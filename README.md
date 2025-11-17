# 🧩 Puzzle Solver

Programa en Python que usa OpenCV para ayudarte a resolver rompecabezas. **Compara tu rompecabezas parcialmente armado con piezas sueltas** y te dice exactamente cuáles encajan y dónde.

## ✨ Novedad: Modo Comparación

**¿Ya tienes el borde armado?** Este programa ahora puede:
- 📸 Analizar una foto de tu rompecabezas actual
- 🔍 Comparar con piezas sueltas que tengas en otra foto
- 🎯 **Decirte qué piezas específicas encajan en los bordes** y con qué probabilidad
- 📊 Mostrarte visualmente las mejores candidatas

## 🎯 Características

- **Modo avanzado**: Compara rompecabezas armado vs piezas sueltas (¡NUEVO!)
- **Detección de forma de bordes**: Identifica pestañas y cavidades automáticamente (¡NUEVO!)
- **Matching por forma + color**: Solo sugiere piezas con formas compatibles (¡NUEVO!)
- **Detección de bordes**: Analiza los bordes del rompecabezas armado
- **Matching inteligente**: Compara colores de bordes para encontrar coincidencias
- **Detección automática de piezas**: Identifica piezas individuales en fotografías
- **Análisis de colores**: Extrae colores dominantes de cada pieza
- **Agrupación por color**: Agrupa piezas similares para facilitar el armado
- **Visualizaciones**: Genera imágenes organizadas de las piezas y sugerencias

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

El programa tiene **DOS MODOS** de operación:

### 🎯 Modo 1: Comparar con rompecabezas armado (RECOMENDADO)

**Este es el modo más útil** - Compara tu rompecabezas parcialmente armado con piezas sueltas para decirte exactamente cuáles encajan:

```bash
python puzzle_solver.py piezas_sueltas.jpg --puzzle-assembled mi_rompecabezas.jpg
```

**Ejemplo real:**
```bash
# Tienes el borde armado y quieres saber qué piezas van en los bordes internos
python puzzle_solver.py grupo_cielo.jpg --puzzle-assembled borde_completo.jpg
```

**Salida:** Te dirá qué piezas específicas encajan mejor en los bordes de tu rompecabezas actual.

### 📊 Modo 2: Solo analizar piezas sueltas

Útil cuando aún no tienes nada armado:

```bash
python puzzle_solver.py piezas_sueltas.jpg
```

### Opciones adicionales

```bash
python puzzle_solver.py piezas.jpg \
  --puzzle-assembled rompecabezas.jpg \
  --output-dir resultados \
  --min-area 500 \
  --tolerance 30 \
  --top-matches 20
```

**Parámetros:**
- `piezas_sueltas.jpg`: Imagen con las piezas sin armar (obligatorio)
- `--puzzle-assembled`: Imagen del rompecabezas parcialmente armado (opcional)
- `--output-dir`: Directorio donde guardar los resultados (default: `output`)
- `--min-area`: Área mínima en píxeles para detectar una pieza (default: 1000)
- `--tolerance`: Tolerancia para agrupar colores similares, 0-255 (default: 50)
- `--top-matches`: Cuántas mejores coincidencias mostrar (default: 15)

## 📊 Resultados

### Modo avanzado (con --puzzle-assembled):

1. **`matches_puzzle_marked.jpg`**: Tu rompecabezas con los bordes marcados en colores
2. **`matches_best_pieces.jpg`**: Las mejores piezas candidatas numeradas con sus scores
3. **Salida en consola**: Lista detallada de qué pieza va dónde

### Modo simple (sin --puzzle-assembled):

1. **`pieces_overview.jpg`**: Vista general de todas las piezas detectadas con sus IDs
2. **`color_groups_*.jpg`**: Múltiples imágenes, una por cada grupo de color
3. **`analysis.json`**: Datos detallados en formato JSON para análisis adicional

## 🎓 Ejemplo de flujo de trabajo

### Para tu rompecabezas de 1000 piezas (YA TIENES EL BORDE ARMADO)

**Método recomendado - Usar el modo avanzado:**

1. **Toma foto de tu borde armado**:
   - Desde arriba, con buena luz
   - Asegúrate que se vea todo el borde completo
   - Guárdala como `borde_armado.jpg`

2. **Agrupa piezas sueltas por color similar** (30-50 piezas):
   - Por ejemplo: todas las piezas azules del cielo
   - Colócalas separadas sobre fondo blanco/negro
   - Toma foto: `piezas_cielo.jpg`

3. **Ejecuta el programa en modo avanzado**:
   ```bash
   python puzzle_solver.py piezas_cielo.jpg \
     --puzzle-assembled borde_armado.jpg \
     --output-dir resultados_cielo
   ```

4. **Revisa los resultados**:
   - El programa te dirá: "Pieza #5 encaja en borde TOP con score 0.892"
   - Abre `matches_best_pieces.jpg` para ver las piezas sugeridas
   - Prueba físicamente las piezas en los lugares indicados

5. **Actualiza tu foto del rompecabezas** a medida que agregas piezas:
   - Arma las piezas que funcionaron
   - Toma nueva foto del progreso
   - Repite el proceso con las piezas restantes

**Método alternativo - Modo simple:**

Útil si aún no tienes nada armado o quieres solo explorar:

```bash
# Analiza piezas sueltas para encontrar similares entre sí
python puzzle_solver.py piezas_variadas.jpg --output-dir resultados
```

## 💡 Consejos

### Para el modo avanzado:
- **Buena foto del rompecabezas armado**: Asegúrate que los bordes estén bien visibles
- **Fondo contrastante**: Coloca el rompecabezas sobre un fondo de color diferente
- **Iluminación uniforme**: Evita sombras en los bordes
- **Actualiza frecuentemente**: Toma nueva foto cada vez que agregas 5-10 piezas
- **Prueba físicamente**: El score es una guía, siempre verifica manualmente

### Para ambos modos:
- **Ajusta `--min-area`**: Si detecta muchos objetos falsos, aumenta este valor
- **Ajusta `--tolerance`**:
  - Valor más bajo (20-30): Grupos más específicos, más grupos
  - Valor más alto (60-80): Grupos más generales, menos grupos
- **Ajusta `--top-matches`**: Muestra más o menos sugerencias según necesites
- **Calidad de imagen**: Una mejor foto = mejores resultados
- **Paciencia**: Este programa es una herramienta de ayuda, no reemplaza el proceso manual

## 🔧 Solución de problemas

### No detecta el rompecabezas armado (modo avanzado)
- Asegúrate que el rompecabezas esté completo en la foto
- Mejora el contraste entre el rompecabezas y el fondo
- Evita sombras fuertes sobre el rompecabezas
- Toma la foto directamente desde arriba

### Las sugerencias no son buenas (modo avanzado)
- Usa piezas de colores similares (pre-agrupa manualmente)
- Asegúrate que las fotos tengan la misma iluminación
- Reduce el número de piezas sueltas por análisis (20-40 es ideal)
- Aumenta `--top-matches` para ver más opciones

### No detecta piezas sueltas
- Verifica que haya buen contraste entre las piezas y el fondo
- Reduce el valor de `--min-area`
- Mejora la iluminación de la foto
- Asegúrate que las piezas no se toquen entre sí

### Detecta demasiados objetos
- Aumenta el valor de `--min-area`
- Limpia el fondo de objetos no deseados
- Usa un fondo más uniforme

### Los grupos de color no son útiles
- Ajusta el valor de `--tolerance`
- Prueba con diferentes valores entre 20 y 80

## 📝 Información técnica

El programa utiliza:
- **Canny Edge Detection**: Para detectar bordes del rompecabezas armado
- **Segmentación de bordes**: Divide bordes en segmentos pequeños para análisis detallado
- **Análisis de curvatura**: Detecta pestañas (tabs) y cavidades (blanks) en cada borde
- **Verificación de compatibilidad**: Solo permite emparejar formas compatibles
- **Comparación de colores**: Algoritmo de similitud basado en distancia euclidiana en espacio RGB
- **Detección de contornos**: Para identificar piezas individuales (cv2.findContours)
- **K-means clustering**: Para encontrar colores dominantes de cada pieza
- **Análisis de bordes por dirección**: Extrae colores de top/bottom/left/right de cada pieza
- **Threshold adaptativo**: Para manejar diferentes condiciones de iluminación

### Detección de forma de bordes:
El programa analiza cada borde de cada pieza para clasificarlo como:
- **Flat (▬)**: Borde recto - lados del rompecabezas
- **Tab (▲)**: Pestaña que sobresale
- **Blank (▼)**: Cavidad que entra

Algoritmo:
1. Extrae puntos del contorno correspondientes a cada borde
2. Calcula desviación estándar para detectar irregularidades
3. Determina si es recto, sobresale o entra basándose en umbrales

### Cómo funciona el matching mejorado:
1. El rompecabezas armado se divide en segmentos de borde (~50px cada uno)
2. Cada pieza suelta se analiza:
   - Extrae colores de sus 4 bordes
   - Detecta forma de cada borde (flat/tab/blank)
3. Se comparan bordes opuestos (top del puzzle con bottom de pieza, etc.)
4. **Verificación de forma**: Solo continúa si las formas son compatibles
   - Tab ↔ Blank: Compatible ✅
   - Flat ↔ Flat: Compatible ✅
   - Tab ↔ Tab: NO compatible ❌
   - Blank ↔ Blank: NO compatible ❌
5. Se calcula score: similitud de color × bonus de forma
   - Piezas con borde flat reciben bonus 2x (son bordes del puzzle)
6. Se ordenan las piezas por mejor score de coincidencia

**Resultado**: Scores más altos y sugerencias más precisas, eliminando falsos positivos por incompatibilidad de forma.

## 🤝 Contribuciones

Este es un proyecto de código abierto. Siéntete libre de mejorarlo y compartir tus modificaciones.

## 📄 Licencia

MIT License - Usa y modifica libremente este código.

---

¡Buena suerte armando tu rompecabezas de 1000 piezas! 🧩✨
