# 🧩 Puzzle Solver

Programa en Python que usa OpenCV para ayudarte a resolver rompecabezas. Analiza piezas, las agrupa por color y sugiere qué piezas podrían encajar juntas.

## 🎯 Características

- **Detección automática de piezas**: Identifica piezas individuales en una fotografía
- **Análisis de colores**: Extrae colores dominantes de cada pieza
- **Agrupación por color**: Agrupa piezas similares para facilitar el armado
- **Sugerencias de coincidencias**: Compara bordes y sugiere piezas que podrían encajar
- **Visualizaciones**: Genera imágenes organizadas de las piezas detectadas

## 📋 Requisitos

- Python 3.7 o superior
- OpenCV
- NumPy

## 🚀 Instalación

1. Clona o descarga este repositorio

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## 📸 Cómo tomar las fotos

Para mejores resultados:

1. **Iluminación uniforme**: Usa buena luz natural o artificial sin sombras
2. **Fondo contrastante**: Coloca las piezas sobre un fondo de color sólido (blanco o negro funciona bien)
3. **Piezas separadas**: Asegúrate de que las piezas no se toquen entre sí
4. **Vista desde arriba**: Toma la foto directamente desde arriba
5. **Enfoque nítido**: Asegúrate de que la imagen esté bien enfocada

### Sugerencia para 1000 piezas

Para un rompecabezas de 1000 piezas:
- Toma fotos de grupos de 20-50 piezas a la vez
- Agrupa primero las piezas por color similares manualmente
- Analiza cada grupo por separado
- El programa te ayudará a refinar y encontrar coincidencias dentro de cada grupo

## 💻 Uso

### Uso básico

```bash
python puzzle_solver.py imagen_piezas.jpg
```

### Opciones avanzadas

```bash
python puzzle_solver.py imagen_piezas.jpg \
  --output-dir resultados \
  --min-area 500 \
  --tolerance 30
```

**Parámetros:**
- `imagen_piezas.jpg`: Ruta a la imagen con las piezas (obligatorio)
- `--output-dir`: Directorio donde guardar los resultados (default: `output`)
- `--min-area`: Área mínima en píxeles para detectar una pieza (default: 1000)
- `--tolerance`: Tolerancia para agrupar colores similares, 0-255 (default: 50)

## 📊 Resultados

El programa genera los siguientes archivos:

1. **`pieces_overview.jpg`**: Vista general de todas las piezas detectadas con sus IDs
2. **`color_groups_*.jpg`**: Múltiples imágenes, una por cada grupo de color
3. **`analysis.json`**: Datos detallados en formato JSON para análisis adicional

## 🎓 Ejemplo de flujo de trabajo

### Para tu rompecabezas de 1000 piezas

1. **Ya tienes el borde armado** ✅

2. **Agrupa las piezas restantes por zona de color**:
   - Cielo / fondo
   - Elementos principales
   - Detalles específicos

3. **Para cada grupo**:
   ```bash
   # Toma una foto del grupo
   python puzzle_solver.py grupo_cielo.jpg --output-dir resultados_cielo

   # Revisa las visualizaciones generadas
   # El programa te mostrará piezas similares
   ```

4. **Usa las sugerencias**:
   - El programa te dirá qué piezas tienen colores de borde similares
   - Prueba físicamente las coincidencias sugeridas
   - Continúa con el siguiente grupo

## 💡 Consejos

- **Ajusta `--min-area`**: Si detecta muchos objetos falsos, aumenta este valor
- **Ajusta `--tolerance`**:
  - Valor más bajo (20-30): Grupos más específicos, más grupos
  - Valor más alto (60-80): Grupos más generales, menos grupos
- **Calidad de imagen**: Una mejor foto = mejores resultados
- **Paciencia**: Este programa es una herramienta de ayuda, no reemplaza el proceso manual

## 🔧 Solución de problemas

### No detecta piezas
- Verifica que haya buen contraste entre las piezas y el fondo
- Reduce el valor de `--min-area`
- Mejora la iluminación de la foto

### Detecta demasiados objetos
- Aumenta el valor de `--min-area`
- Limpia el fondo de objetos no deseados

### Los grupos de color no son útiles
- Ajusta el valor de `--tolerance`
- Prueba con diferentes valores entre 20 y 80

## 📝 Información técnica

El programa utiliza:
- **Detección de contornos**: Para identificar piezas individuales
- **K-means clustering**: Para encontrar colores dominantes
- **Análisis de bordes**: Para comparar y sugerir coincidencias
- **Threshold adaptativo**: Para manejar diferentes condiciones de iluminación

## 🤝 Contribuciones

Este es un proyecto de código abierto. Siéntete libre de mejorarlo y compartir tus modificaciones.

## 📄 Licencia

MIT License - Usa y modifica libremente este código.

---

¡Buena suerte armando tu rompecabezas de 1000 piezas! 🧩✨
