# Modelo Simplificado de Balance Hídrico del Embalse en Cerro Catedral

Este modelo simula la evolución temporal del nivel de un embalse utilizado para generar nieve artificial, considerando aportes hídricos desde un arroyo, consumo para producción de nieve y descargas naturales. La dinámica depende de la temperatura de bulbo húmedo (`Tw`), obtenida a partir de datos atmosféricos.

---

## Supuestos principales

- El embalse no recibe precipitaciones directas ni tiene pérdidas por evaporación.
- La entrada de agua (`A1`) proviene del Arroyo La Cascada y depende de la temperatura `Tw`, modelada mediante una **sigmoide** (aunque puede usarse también una función exponencial).
- El consumo de agua (`N`) para generar nieve depende también de `Tw`, siendo alto a bajas temperaturas y nulo para `Tw > 0°C`.
- La salida de agua (`A2`) depende del volumen embalsado, pero tiene una cota mínima, representando un caudal de "reserva moral" destinado a las personas o al ecosistema aguas abajo.
- El modelo se resuelve mediante el método de **Euler explícito**, con paso diario.

---

## Cálculo de constantes

- **`N_max = 6000 m³/día`**: estimado suponiendo producción de 10 cm de nieve sobre 10 hectáreas por día, con densidad 0.6 (comparada con el agua líquida).

  > 10 ha × 0.1 m × 10000 m²/ha × 0.6 ≈ 6000 m³

- **`A1_max = 10800 m³/día`**: representa un valor medio de escurrimiento del Arroyo La Cascada, que puede cambiar significativamente con lluvias, deshielo, o sequías.

- **`R_max = 125000 m³`**: volumen máximo del embalse, tomado como valor de referencia técnica/local.

---

## Aspectos por mejorar

### Ingreso de agua (`A1`)
Actualmente se modela solo en función de `Tw`, pero en la práctica también depende de:

- la **disponibilidad real de agua** (e.g. deshielo, nivel freático, precipitación previa),
- si hay alguna **bomba** o sistema de extracción artificial desde el lago,
- o si la montaña está seca (condiciones de verano o fin de temporada).

Podría implementarse un esquema de acumulación/agotamiento del escurrimiento disponible, o incorporar una función multivariada (e.g. Tw, nieve acumulada, precipitación, NDVI, etc.).

### Salida (`A2`)
Además de depender del nivel del embalse, se establece un **mínimo moral (`A2_min`)** que representa el caudal necesario para sostener el uso humano y ecológico del agua aguas abajo, incluso si el embalse se encuentra bajo.

---
...

## Archivos requeridos

El repositorio ya incluye todos los archivos necesarios para correr el modelo:

- `modelo_reservorio.py`: script principal con el modelo completo
- `Tw_desde_T_y_RH.csv`: archivo de entrada con datos reales de temperatura de bulbo húmedo
  - Columnas:
    - `t`: fechas (formato compatible con `pandas.to_datetime()`)
    - `Tw`: temperatura de bulbo húmedo [°C]

Este archivo `.csv` se proporciona como ejemplo y permite probar directamente el funcionamiento del modelo.

## Requisitos

- Python 3.8+
- Bibliotecas:
  - `numpy`
  - `pandas`
  - `matplotlib`

Instalación rápida:

```bash
pip install numpy pandas matplotlib

## Cómo ejecutar el modelo?

1. Cloná este repositorio o descargalo como ZIP.
2. Asegurate de tener el archivo `Tw_desde_T_y_RH.csv` en el mismo directorio que el script (ya está incluido).
3. Corré el script principal:

```bash
python modelo_reservorio.py
