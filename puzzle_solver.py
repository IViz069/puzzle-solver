#!/usr/bin/env python3
"""
Puzzle Solver - Programa para ayudar a resolver rompecabezas usando OpenCV
Analiza piezas, las agrupa por color y sugiere coincidencias

Modos de uso:
1. Modo simple: Analiza solo piezas sueltas
2. Modo avanzado: Compara rompecabezas armado con piezas sueltas
"""

import cv2
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
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


class PuzzleEdge:
    """Representa un borde del rompecabezas armado"""

    def __init__(self, position: Tuple[int, int], direction: str, color: Tuple, length: int):
        self.position = position  # (x, y) donde está el borde
        self.direction = direction  # 'top', 'bottom', 'left', 'right'
        self.color = color  # color promedio del borde
        self.length = length  # longitud del segmento de borde


class AssembledPuzzle:
    """Analiza un rompecabezas parcialmente armado"""

    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = None
        self.edges: List[PuzzleEdge] = []
        self.contour = None
        self.bbox = None

    def analyze(self):
        """Analiza el rompecabezas armado y detecta sus bordes"""
        print(f"🧩 Analizando rompecabezas armado: {self.image_path}")
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError(f"No se pudo cargar la imagen: {self.image_path}")

        # Convertir a escala de grises
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Aplicar blur para reducir ruido
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detectar bordes con Canny
        edges = cv2.Canny(blurred, 50, 150)

        # Encontrar el contorno del rompecabezas
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("⚠️  No se detectó el contorno del rompecabezas")
            return False

        # Tomar el contorno más grande (debería ser el rompecabezas)
        self.contour = max(contours, key=cv2.contourArea)
        self.bbox = cv2.boundingRect(self.contour)

        print(f"✅ Rompecabezas detectado")

        # Extraer información de los bordes
        self._extract_edge_info()

        return True

    def _extract_edge_info(self):
        """Extrae información de color de los bordes del rompecabezas"""
        if self.contour is None:
            return

        x, y, w, h = self.bbox
        edge_width = 20  # Ancho de la franja para analizar

        # Analizar cada lado del rompecabezas
        # Top edge
        if y > edge_width:
            top_region = self.image[y-edge_width:y+edge_width, x:x+w]
            top_color = cv2.mean(top_region)[:3]
            self.edges.append(PuzzleEdge((x + w//2, y), 'top', top_color, w))

        # Bottom edge
        if y + h < self.image.shape[0] - edge_width:
            bottom_region = self.image[y+h-edge_width:y+h+edge_width, x:x+w]
            bottom_color = cv2.mean(bottom_region)[:3]
            self.edges.append(PuzzleEdge((x + w//2, y+h), 'bottom', bottom_color, w))

        # Left edge
        if x > edge_width:
            left_region = self.image[y:y+h, x-edge_width:x+edge_width]
            left_color = cv2.mean(left_region)[:3]
            self.edges.append(PuzzleEdge((x, y + h//2), 'left', left_color, h))

        # Right edge
        if x + w < self.image.shape[1] - edge_width:
            right_region = self.image[y:y+h, x+w-edge_width:x+w+edge_width]
            right_color = cv2.mean(right_region)[:3]
            self.edges.append(PuzzleEdge((x+w, y + h//2), 'right', right_color, h))

        print(f"📏 Bordes detectados: {len(self.edges)}")
        for edge in self.edges:
            print(f"   Borde {edge.direction}: color promedio RGB{tuple(int(c) for c in edge.color)}")

    def get_edge_segments(self, segment_size=50):
        """Divide los bordes en segmentos más pequeños para análisis detallado"""
        segments = []

        x, y, w, h = self.bbox
        edge_width = 15

        # Segmentar borde superior
        for i in range(0, w, segment_size):
            x_start = x + i
            x_end = min(x + i + segment_size, x + w)
            region = self.image[max(0, y-edge_width):y+edge_width, x_start:x_end]
            if region.size > 0:
                color = cv2.mean(region)[:3]
                segments.append({
                    'position': (x_start + (x_end - x_start)//2, y),
                    'direction': 'top',
                    'color': color,
                    'width': x_end - x_start
                })

        # Segmentar borde inferior
        for i in range(0, w, segment_size):
            x_start = x + i
            x_end = min(x + i + segment_size, x + w)
            region = self.image[y+h-edge_width:min(self.image.shape[0], y+h+edge_width), x_start:x_end]
            if region.size > 0:
                color = cv2.mean(region)[:3]
                segments.append({
                    'position': (x_start + (x_end - x_start)//2, y+h),
                    'direction': 'bottom',
                    'color': color,
                    'width': x_end - x_start
                })

        # Segmentar borde izquierdo
        for i in range(0, h, segment_size):
            y_start = y + i
            y_end = min(y + i + segment_size, y + h)
            region = self.image[y_start:y_end, max(0, x-edge_width):x+edge_width]
            if region.size > 0:
                color = cv2.mean(region)[:3]
                segments.append({
                    'position': (x, y_start + (y_end - y_start)//2),
                    'direction': 'left',
                    'color': color,
                    'width': y_end - y_start
                })

        # Segmentar borde derecho
        for i in range(0, h, segment_size):
            y_start = y + i
            y_end = min(y + i + segment_size, y + h)
            region = self.image[y_start:y_end, x+w-edge_width:min(self.image.shape[1], x+w+edge_width)]
            if region.size > 0:
                color = cv2.mean(region)[:3]
                segments.append({
                    'position': (x+w, y_start + (y_end - y_start)//2),
                    'direction': 'right',
                    'color': color,
                    'width': y_end - y_start
                })

        return segments


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

    def find_pieces_for_puzzle(self, assembled_puzzle: AssembledPuzzle, top_n=10):
        """Encuentra qué piezas sueltas encajan mejor con el rompecabezas armado"""
        print(f"\n🔍 Buscando piezas que encajen con el rompecabezas armado...")

        # Obtener segmentos de los bordes del rompecabezas
        puzzle_segments = assembled_puzzle.get_edge_segments(segment_size=50)

        print(f"📊 Analizando {len(puzzle_segments)} segmentos de borde del rompecabezas")
        print(f"📊 Comparando con {len(self.pieces)} piezas sueltas\n")

        # Mapeo de direcciones opuestas
        opposite = {
            'top': 'bottom',
            'bottom': 'top',
            'left': 'right',
            'right': 'left'
        }

        # Encontrar mejores coincidencias
        matches = []

        for piece in self.pieces:
            best_match_score = 0
            best_segment = None
            best_edge = None

            # Comparar cada borde de la pieza con cada segmento del puzzle
            for edge_name, edge_color in piece.edge_colors.items():
                for segment in puzzle_segments:
                    # Solo comparar bordes compatibles (opuestos)
                    if opposite[segment['direction']] == edge_name:
                        # Calcular similitud de color
                        puzzle_color = segment['color']
                        distance = np.sqrt(sum((c1 - c2) ** 2
                                             for c1, c2 in zip(edge_color, puzzle_color)))
                        similarity = 1 / (1 + distance)

                        if similarity > best_match_score:
                            best_match_score = similarity
                            best_segment = segment
                            best_edge = edge_name

            if best_match_score > 0:
                matches.append({
                    'piece_id': piece.id,
                    'score': best_match_score,
                    'piece_edge': best_edge,
                    'puzzle_position': best_segment['position'] if best_segment else None,
                    'puzzle_direction': best_segment['direction'] if best_segment else None
                })

        # Ordenar por score
        matches.sort(key=lambda x: x['score'], reverse=True)

        return matches[:top_n]

    def visualize_puzzle_matches(self, assembled_puzzle: AssembledPuzzle,
                                 matches: List[Dict], output_path: str):
        """Crea una visualización mostrando las mejores piezas para el puzzle armado"""
        print(f"🖼️  Creando visualización de coincidencias...")

        # Copiar la imagen del rompecabezas armado
        result = assembled_puzzle.image.copy()

        # Dibujar el contorno del puzzle
        if assembled_puzzle.contour is not None:
            cv2.drawContours(result, [assembled_puzzle.contour], -1, (0, 255, 0), 3)

        # Marcar los bordes
        x, y, w, h = assembled_puzzle.bbox
        # Top
        cv2.line(result, (x, y), (x+w, y), (255, 0, 0), 2)
        # Bottom
        cv2.line(result, (x, y+h), (x+w, y+h), (255, 0, 0), 2)
        # Left
        cv2.line(result, (x, y), (x, y+h), (255, 0, 0), 2)
        # Right
        cv2.line(result, (x+w, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imwrite(output_path.replace('.jpg', '_puzzle_marked.jpg'), result)

        # Crear visualización de las mejores piezas
        max_show = min(12, len(matches))
        cols = 4
        rows = (max_show + cols - 1) // cols
        cell_size = 200

        pieces_img = np.ones((rows * cell_size, cols * cell_size, 3), dtype=np.uint8) * 255

        for idx, match in enumerate(matches[:max_show]):
            piece = self.pieces[match['piece_id']]
            row = idx // cols
            col = idx % cols

            # Redimensionar pieza
            piece_img = piece.image
            h_p, w_p = piece_img.shape[:2]

            scale = min((cell_size - 40) / w_p, (cell_size - 40) / h_p)
            new_w = int(w_p * scale)
            new_h = int(h_p * scale)

            if new_w > 0 and new_h > 0:
                resized = cv2.resize(piece_img, (new_w, new_h))

                y_offset = row * cell_size + (cell_size - new_h) // 2
                x_offset = col * cell_size + (cell_size - new_w) // 2

                mask = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) > 10
                pieces_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w][mask] = resized[mask]

                # Agregar información
                text_y = row * cell_size + cell_size - 60
                cv2.putText(pieces_img, f"Pieza {piece.id}",
                           (col * cell_size + 10, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.putText(pieces_img, f"Score: {match['score']:.3f}",
                           (col * cell_size + 10, text_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 0), 1)
                cv2.putText(pieces_img, f"{match['puzzle_direction']}",
                           (col * cell_size + 10, text_y + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 0, 0), 1)

        cv2.imwrite(output_path.replace('.jpg', '_best_pieces.jpg'), pieces_img)
        print(f"✅ Visualizaciones guardadas:")
        print(f"   - {output_path.replace('.jpg', '_puzzle_marked.jpg')}")
        print(f"   - {output_path.replace('.jpg', '_best_pieces.jpg')}")

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
        description='Puzzle Solver - Ayuda a resolver rompecabezas usando visión por computadora',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modos de uso:

1. Modo simple (solo piezas sueltas):
   python puzzle_solver.py piezas_sueltas.jpg

2. Modo avanzado (rompecabezas armado + piezas sueltas):
   python puzzle_solver.py piezas_sueltas.jpg --puzzle-assembled rompecabezas.jpg

El modo avanzado te dirá qué piezas específicas encajan mejor con tu rompecabezas actual.
        """
    )
    parser.add_argument('pieces_image',
                       help='Ruta a la imagen con las piezas sueltas del rompecabezas')
    parser.add_argument('--puzzle-assembled',
                       help='Ruta a la imagen del rompecabezas parcialmente armado')
    parser.add_argument('--output-dir', default='output',
                       help='Directorio para guardar resultados (default: output)')
    parser.add_argument('--min-area', type=int, default=1000,
                       help='Área mínima para detectar una pieza (default: 1000)')
    parser.add_argument('--tolerance', type=int, default=50,
                       help='Tolerancia para agrupación de colores (default: 50)')
    parser.add_argument('--top-matches', type=int, default=15,
                       help='Número de mejores coincidencias a mostrar (default: 15)')

    args = parser.parse_args()

    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("🧩 Puzzle Solver - Iniciando análisis...")
    print("=" * 60)

    # Determinar modo de operación
    mode = "avanzado" if args.puzzle_assembled else "simple"
    print(f"📋 Modo: {mode.upper()}")
    print("=" * 60)

    # Crear solver y detectar piezas sueltas
    solver = PuzzleSolver()
    n_pieces = solver.detect_pieces(args.pieces_image, min_area=args.min_area)

    if n_pieces == 0:
        print("❌ No se detectaron piezas sueltas. Intenta con una imagen diferente.")
        return

    # Modo avanzado: comparar con rompecabezas armado
    if args.puzzle_assembled:
        print("\n" + "=" * 60)
        print("🎯 MODO AVANZADO: Comparando con rompecabezas armado")
        print("=" * 60)

        # Analizar rompecabezas armado
        puzzle = AssembledPuzzle(args.puzzle_assembled)
        if not puzzle.analyze():
            print("⚠️  No se pudo analizar el rompecabezas armado. Continuando en modo simple...")
            args.puzzle_assembled = None
        else:
            # Encontrar las mejores coincidencias
            matches = solver.find_pieces_for_puzzle(puzzle, top_n=args.top_matches)

            if matches:
                print("\n" + "=" * 60)
                print(f"🎯 TOP {len(matches)} PIEZAS QUE MEJOR ENCAJAN:")
                print("=" * 60)

                for i, match in enumerate(matches, 1):
                    piece = solver.pieces[match['piece_id']]
                    print(f"\n{i}. 🧩 Pieza #{match['piece_id']}")
                    print(f"   ✨ Score: {match['score']:.3f}")
                    print(f"   📍 Encajaría en borde: {match['puzzle_direction']}")
                    print(f"   🔄 Lado de pieza: {match['piece_edge']}")
                    colors = [int(c) for c in piece.dominant_colors[0]] if piece.dominant_colors else [0, 0, 0]
                    print(f"   🎨 Color dominante: RGB{tuple(colors)}")

                # Crear visualización
                solver.visualize_puzzle_matches(
                    puzzle, matches,
                    str(output_dir / 'matches.jpg')
                )

                print("\n" + "=" * 60)
                print("✅ Análisis completado!")
                print(f"📁 Resultados guardados en: {output_dir}")
                print("\n📋 Archivos generados:")
                print("   • matches_puzzle_marked.jpg - Rompecabezas con bordes marcados")
                print("   • matches_best_pieces.jpg - Mejores piezas candidatas")
                print("\n💡 Siguiente paso:")
                print("   Prueba físicamente las piezas sugeridas en los bordes indicados")
            else:
                print("\n⚠️  No se encontraron coincidencias significativas")

    # Modo simple o complementario: análisis de piezas sueltas
    if not args.puzzle_assembled or mode == "simple":
        print("\n" + "=" * 60)
        if mode == "avanzado":
            print("📊 ANÁLISIS COMPLEMENTARIO: Agrupación de piezas sueltas")
        else:
            print("📊 ANÁLISIS: Agrupación de piezas sueltas")
        print("=" * 60)

        # Agrupar por color
        solver.group_by_color(tolerance=args.tolerance)

        # Guardar visualizaciones
        solver.visualize_pieces(str(output_dir / 'pieces_overview.jpg'))
        solver.visualize_color_groups(str(output_dir / 'color_groups.jpg'))

        # Guardar análisis
        solver.save_analysis(str(output_dir / 'analysis.json'))

        # Mostrar algunas sugerencias entre piezas
        print("\n" + "=" * 60)
        print("💡 Sugerencias entre piezas sueltas (top 3):")
        print("=" * 60)

        for i in range(min(3, len(solver.pieces))):
            matches = solver.find_matching_pieces(i, top_n=3)
            if matches:
                print(f"\n🧩 Pieza {i} - Posibles vecinas:")
                for match_id, similarity in matches:
                    print(f"   → Pieza {match_id}: {similarity:.3f} similitud")

        print("\n" + "=" * 60)
        print("✅ Análisis completado!")
        print(f"📁 Resultados guardados en: {output_dir}")
        print("\n📋 Archivos generados:")
        print("   • pieces_overview.jpg - Vista de todas las piezas")
        print("   • color_groups_*.jpg - Piezas agrupadas por color")
        print("   • analysis.json - Datos detallados para análisis")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
