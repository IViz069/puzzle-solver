#!/usr/bin/env python3
"""
Puzzle Solver - Programa para ayudar a resolver rompecabezas usando OpenCV
Analiza piezas, las agrupa por color y sugiere coincidencias
"""

import cv2
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
from typing import List, Tuple, Dict
import argparse


class PuzzlePiece:
    """Representa una pieza individual del rompecabezas"""

    def __init__(self, id: int, contour, image, bbox):
        self.id = id
        self.contour = contour
        self.image = image
        self.bbox = bbox  # (x, y, w, h)
        self.dominant_colors = []
        self.edge_colors = {'top': [], 'bottom': [], 'left': [], 'right': []}
        self.area = cv2.contourArea(contour)

    def extract_colors(self, num_colors=3):
        """Extrae los colores dominantes de la pieza"""
        pixels = self.image.reshape(-1, 3)
        pixels = np.float32(pixels)

        # Usar K-means para encontrar colores dominantes
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10,
                                        cv2.KMEANS_PP_CENTERS)

        centers = np.uint8(centers)
        self.dominant_colors = centers.tolist()
        return self.dominant_colors

    def extract_edge_colors(self):
        """Extrae colores de cada borde de la pieza"""
        h, w = self.image.shape[:2]
        edge_width = max(1, min(h, w) // 10)

        # Top
        top_edge = self.image[:edge_width, :]
        self.edge_colors['top'] = self._get_avg_color(top_edge)

        # Bottom
        bottom_edge = self.image[-edge_width:, :]
        self.edge_colors['bottom'] = self._get_avg_color(bottom_edge)

        # Left
        left_edge = self.image[:, :edge_width]
        self.edge_colors['left'] = self._get_avg_color(left_edge)

        # Right
        right_edge = self.image[:, -edge_width:]
        self.edge_colors['right'] = self._get_avg_color(right_edge)

        return self.edge_colors

    def _get_avg_color(self, region):
        """Calcula el color promedio de una región"""
        return cv2.mean(region)[:3]


class PuzzleSolver:
    """Clase principal para resolver rompecabezas"""

    def __init__(self):
        self.pieces: List[PuzzlePiece] = []
        self.color_groups: Dict[str, List[int]] = defaultdict(list)

    def detect_pieces(self, image_path: str, min_area=1000):
        """Detecta piezas individuales en una imagen"""
        print(f"📷 Cargando imagen: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")

        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Aplicar threshold adaptativo
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Encontrar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        print(f"🔍 Contornos encontrados: {len(contours)}")

        # Filtrar y procesar contornos
        piece_id = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # Obtener bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Extraer la pieza
            piece_img = image[y:y+h, x:x+w].copy()

            # Crear máscara para la pieza
            mask = np.zeros((h, w), dtype=np.uint8)
            contour_shifted = contour - [x, y]
            cv2.drawContours(mask, [contour_shifted], -1, 255, -1)

            # Aplicar máscara
            piece_img = cv2.bitwise_and(piece_img, piece_img, mask=mask)

            # Crear objeto PuzzlePiece
            piece = PuzzlePiece(piece_id, contour, piece_img, (x, y, w, h))
            piece.extract_colors()
            piece.extract_edge_colors()

            self.pieces.append(piece)
            piece_id += 1

        print(f"✅ Piezas detectadas: {len(self.pieces)}")
        return len(self.pieces)

    def group_by_color(self, tolerance=50):
        """Agrupa piezas por similitud de color"""
        print(f"🎨 Agrupando piezas por color...")

        self.color_groups.clear()

        # Crear grupos basados en colores dominantes
        for piece in self.pieces:
            if not piece.dominant_colors:
                continue

            # Usar el color dominante principal
            main_color = piece.dominant_colors[0]
            color_key = self._color_to_group(main_color, tolerance)
            self.color_groups[color_key].append(piece.id)

        print(f"✅ Grupos de color creados: {len(self.color_groups)}")
        for color, pieces in self.color_groups.items():
            print(f"   Color {color}: {len(pieces)} piezas")

        return self.color_groups

    def _color_to_group(self, color, tolerance):
        """Convierte un color a una clave de grupo con tolerancia"""
        # Redondear cada componente al múltiplo más cercano de tolerance
        r, g, b = color
        r = (r // tolerance) * tolerance
        g = (g // tolerance) * tolerance
        b = (b // tolerance) * tolerance
        return f"RGB({r},{g},{b})"

    def find_matching_pieces(self, piece_id: int, top_n=5):
        """Encuentra las piezas que mejor coinciden con una pieza dada"""
        if piece_id >= len(self.pieces):
            return []

        target_piece = self.pieces[piece_id]
        matches = []

        for piece in self.pieces:
            if piece.id == piece_id:
                continue

            # Calcular similitud basada en colores de bordes
            similarity = self._calculate_edge_similarity(target_piece, piece)
            matches.append((piece.id, similarity))

        # Ordenar por similitud (mayor primero)
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_n]

    def _calculate_edge_similarity(self, piece1: PuzzlePiece, piece2: PuzzlePiece):
        """Calcula la similitud entre dos piezas basándose en sus bordes"""
        total_similarity = 0
        comparisons = 0

        # Comparar bordes opuestos (top de 1 con bottom de 2, etc.)
        edge_pairs = [
            ('top', 'bottom'),
            ('bottom', 'top'),
            ('left', 'right'),
            ('right', 'left')
        ]

        for edge1, edge2 in edge_pairs:
            color1 = piece1.edge_colors[edge1]
            color2 = piece2.edge_colors[edge2]

            # Calcular distancia euclidiana en el espacio de color
            distance = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))

            # Convertir distancia a similitud (menor distancia = mayor similitud)
            similarity = 1 / (1 + distance)
            total_similarity += similarity
            comparisons += 1

        return total_similarity / comparisons if comparisons > 0 else 0

    def visualize_pieces(self, output_path: str, max_pieces=50):
        """Crea una visualización de las piezas detectadas"""
        print(f"🖼️  Creando visualización...")

        # Calcular grid size
        n_pieces = min(len(self.pieces), max_pieces)
        cols = int(np.ceil(np.sqrt(n_pieces)))
        rows = int(np.ceil(n_pieces / cols))

        # Tamaño de cada celda
        cell_size = 150

        # Crear imagen de salida
        output = np.ones((rows * cell_size, cols * cell_size, 3), dtype=np.uint8) * 255

        for idx, piece in enumerate(self.pieces[:max_pieces]):
            row = idx // cols
            col = idx % cols

            # Redimensionar pieza
            piece_img = piece.image
            h, w = piece_img.shape[:2]

            # Calcular escala para ajustar a la celda
            scale = min((cell_size - 20) / w, (cell_size - 20) / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            if new_w > 0 and new_h > 0:
                resized = cv2.resize(piece_img, (new_w, new_h))

                # Centrar en la celda
                y_offset = row * cell_size + (cell_size - new_h) // 2
                x_offset = col * cell_size + (cell_size - new_w) // 2

                # Copiar (solo donde la imagen no es negra)
                mask = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) > 10
                output[y_offset:y_offset+new_h, x_offset:x_offset+new_w][mask] = \
                    resized[mask]

                # Agregar ID
                cv2.putText(output, str(piece.id),
                           (col * cell_size + 5, row * cell_size + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imwrite(output_path, output)
        print(f"✅ Visualización guardada en: {output_path}")

    def visualize_color_groups(self, output_path: str):
        """Crea una visualización de los grupos de color"""
        print(f"🎨 Creando visualización de grupos de color...")

        # Crear una imagen para cada grupo
        cell_size = 150
        max_pieces_per_group = 20

        for color_key, piece_ids in self.color_groups.items():
            n_pieces = min(len(piece_ids), max_pieces_per_group)
            if n_pieces == 0:
                continue

            cols = int(np.ceil(np.sqrt(n_pieces)))
            rows = int(np.ceil(n_pieces / cols))

            group_img = np.ones((rows * cell_size, cols * cell_size, 3),
                               dtype=np.uint8) * 255

            for idx, piece_id in enumerate(piece_ids[:max_pieces_per_group]):
                piece = self.pieces[piece_id]
                row = idx // cols
                col = idx % cols

                piece_img = piece.image
                h, w = piece_img.shape[:2]

                scale = min((cell_size - 20) / w, (cell_size - 20) / h)
                new_w = int(w * scale)
                new_h = int(h * scale)

                if new_w > 0 and new_h > 0:
                    resized = cv2.resize(piece_img, (new_w, new_h))

                    y_offset = row * cell_size + (cell_size - new_h) // 2
                    x_offset = col * cell_size + (cell_size - new_w) // 2

                    mask = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) > 10
                    group_img[y_offset:y_offset+new_h,
                             x_offset:x_offset+new_w][mask] = resized[mask]

                    cv2.putText(group_img, str(piece.id),
                               (col * cell_size + 5, row * cell_size + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Guardar grupo
            safe_color = color_key.replace('(', '').replace(')', '').replace(',', '_')
            group_path = output_path.replace('.jpg', f'_{safe_color}.jpg')
            cv2.imwrite(group_path, group_img)
            print(f"   Grupo {color_key}: {group_path}")

    def save_analysis(self, output_path: str):
        """Guarda el análisis en formato JSON"""
        data = {
            'total_pieces': len(self.pieces),
            'color_groups': {k: v for k, v in self.color_groups.items()},
            'pieces': []
        }

        for piece in self.pieces:
            piece_data = {
                'id': piece.id,
                'bbox': piece.bbox,
                'area': float(piece.area),
                'dominant_colors': piece.dominant_colors,
                'edge_colors': piece.edge_colors
            }
            data['pieces'].append(piece_data)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"💾 Análisis guardado en: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Puzzle Solver - Ayuda a resolver rompecabezas usando visión por computadora'
    )
    parser.add_argument('image', help='Ruta a la imagen con las piezas del rompecabezas')
    parser.add_argument('--output-dir', default='output',
                       help='Directorio para guardar resultados (default: output)')
    parser.add_argument('--min-area', type=int, default=1000,
                       help='Área mínima para detectar una pieza (default: 1000)')
    parser.add_argument('--tolerance', type=int, default=50,
                       help='Tolerancia para agrupación de colores (default: 50)')

    args = parser.parse_args()

    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("🧩 Puzzle Solver - Iniciando análisis...")
    print("=" * 60)

    # Crear solver
    solver = PuzzleSolver()

    # Detectar piezas
    n_pieces = solver.detect_pieces(args.image, min_area=args.min_area)

    if n_pieces == 0:
        print("❌ No se detectaron piezas. Intenta con una imagen diferente.")
        return

    # Agrupar por color
    solver.group_by_color(tolerance=args.tolerance)

    # Guardar visualizaciones
    solver.visualize_pieces(str(output_dir / 'pieces_overview.jpg'))
    solver.visualize_color_groups(str(output_dir / 'color_groups.jpg'))

    # Guardar análisis
    solver.save_analysis(str(output_dir / 'analysis.json'))

    # Mostrar sugerencias para las primeras piezas
    print("\n" + "=" * 60)
    print("💡 Sugerencias de coincidencias (top 5):")
    print("=" * 60)

    for i in range(min(5, len(solver.pieces))):
        matches = solver.find_matching_pieces(i)
        print(f"\n🧩 Pieza {i} - Mejores coincidencias:")
        for match_id, similarity in matches[:5]:
            print(f"   → Pieza {match_id}: {similarity:.3f} similitud")

    print("\n" + "=" * 60)
    print("✅ Análisis completado!")
    print(f"📁 Resultados guardados en: {output_dir}")
    print("\nConsejos:")
    print("  • Revisa 'pieces_overview.jpg' para ver todas las piezas detectadas")
    print("  • Revisa 'color_groups_*.jpg' para ver piezas agrupadas por color")
    print("  • Usa 'analysis.json' para análisis programático")


if __name__ == '__main__':
    main()
