# Guion oral — PaleoVox GUI: una historia de datos que no existían

**XIX Congreso Nacional de Paleontología · La Paz, BCS**
**Duración objetivo: 10 min (con margen de 40 s) · 15 láminas**

> Consejo: no leer las láminas. Usar estas líneas como guía; cada bloque indica lo esencial que debe escucharse.

---

## Lámina 1 · Portada (20 s)

Buenos días/tardes. Mi nombre es Alan Amaro, de la Facultad de Ciencias de la UNAM, y hoy les voy a contar una historia: la de un problema que nos encontramos al intentar entrenar inteligencia artificial con fósiles… y la herramienta visual con la que lo resolvimos. Este trabajo es en conjunto con Ángel Ángeles, Jesús Alberto Díaz y Jair Barrientos, del Instituto de Geología.

*Transición:* «Todo empieza con un sueño.»

---

## Lámina 2 · El sueño: una IA que ayudara a reconstruir fósiles (50 s)

La idea era: un modelo de aprendizaje profundo que pudiera completar, clasificar o comparar fósiles incompletos. Pero para aprender, la IA necesita muchísimos ejemplos, y el registro fósil es exactamente lo contrario: cada género está representado por unos pocos ejemplares; son frágiles, no se pueden manipular una y otra vez; casi siempre llegan incompletos por compactación, erosión o fractura; y digitalizarlos cuesta equipo, tiempo y dinero.

El resultado: los modelos ven pocos ejemplos limpios y fallan justo con lo más común en el campo, que son los fósiles dañados.

*Transición:* «Si no hay datos… ¿por qué no fabricarlos?»

---

## Lámina 3 · No había datos. Decidimos fabricarlos. (50 s)

Así nació PaleoVox: un laboratorio virtual de tafonomía. Toma un modelo digital 3D y genera variantes dañadas aplicando los procesos que los fósiles realmente sufren: compactación por sedimento, erosión, rotación y fractura. Con un solo espécimen podemos alimentar miles de ejemplos de entrenamiento, sin tocar nunca el fósil original.

*Señalar las dos imágenes:* el escaneo original y, junto a él, una variante generada por los algoritmos de PaleoVox.

*Transición:* «Pero aquí apareció un muro inesperado.»

---

## Lámina 4 · El muro: la herramienta hablaba en código (35 s)

PaleoVox es una biblioteca de programación: potente, pero cada daño exige escribir código. Y la barrera no era científica, era técnica: quien mejor conoce el fósil no siempre programa. Por eso construimos PaleoVox GUI: exactamente los mismos algoritmos, pero en ventanas. Arrastrar un archivo, ajustar parámetros y ver resultados.

*Transición:* «Veamos la interfaz.»

---

## Lámina 5 · PaleoVox GUI de un vistazo (40 s)

Esta es la aplicación. A la izquierda, la zona de arrastre y la información del modelo. A la derecha, cinco pestañas que ordenan todo el trabajo: pipeline, reconstrucción, vista, t-SNE y perspectivas 2D. En esta captura corrimos el flujo completo sobre un cráneo: después de dilatar, deformar, erosionar y fracturar quedaron 91,203 vóxeles ocupados. Todos los controles tienen valores por defecto y se pueden reiniciar en un clic.

---

## Lámina 6 · El flujo completo, sin escribir código (45 s)

El uso es una cadena sencilla: arrastrar el archivo (.ply, .obj o .npy), convertir la malla a vóxeles, aplicar los daños —que se acumulan uno tras otro—, reconstruir la malla y exportar. Todo corre localmente en la computadora: sin nube, sin cuentas y sin que los datos salgan del equipo.

---

## Lámina 7 · Cuatro daños tafonómicos (35 s)

Estos son los cuatro procesos que podemos simular y combinar, cada uno con sus parámetros: deformación o compactación, erosión, rotación y fractura. En las siguientes cuatro láminas voy a mostrar cada daño en tres representaciones: la rejilla de vóxeles, la malla 3D reconstruida y una perspectiva 2D que compara el original contra la variante.

---

## Lámina 8 · Deformación: el peso de los sedimentos (35 s)

Compactación con factor 0.65 en el eje Z. En los vóxeles se ve el aplastamiento; la malla reconstruida mantiene el aspecto de un fósil real; y en la perspectiva 2D, el azul es el original y el rojo la variante: el contorno externo se conserva, pero el volumen pierde altura. Es el clásico fósil aplastado por el sedimento.

---

## Lámina 9 · Erosión: lo que el tiempo se llevó (35 s)

Aquí truncamos el volumen en el eje X. La rejilla muestra la sección vacía; la malla reconstruida es un fósil incompleto, cerrado y listo para visualizar; y la perspectiva 2D deja ver un corte recto en uno de los extremos. Así se ven muchos de los ejemplares que llegan al laboratorio.

---

## Lámina 10 · Rotación: el mismo fósil, otra orientación (30 s)

La rotación no cambia la forma: gira el espécimen alrededor de su centro. Es clave para que los modelos aprendan a reconocer un fósil hallado en cualquier posición, como ocurre en campo.

---

## Lámina 11 · Fractura: las grietas del terreno (35 s)

La fractura propaga grietas estocásticas que retiran material a su paso. En los vóxeles se ve el volumen en gris con las grietas en rojo; en la malla, el espécimen agrietado; y en el 2D, la grieta roja sobre la silueta original. Simula los fósiles rotos y parcialmente perdidos del registro.

---

## Lámina 12 · Un fósil con historia: los daños combinados (45 s)

Lo interesante es combinar los daños: aquí aplicamos compactación 0.85, erosión 0.45 y fractura con propagación 14, acumulados paso a paso en la GUI. La reconstrucción devuelve una malla cerrada, «watertight», que conserva la escala y la posición del modelo original. Es, literalmente, un fósil que sufrió varias historias… y del que además generamos los datos de entrenamiento.

---

## Lámina 13 · ¿El daño es creíble? (40 s)

No basta con generar variantes: hay que verificarlas. La pestaña de t-SNE mapea cada vóxel a un espacio 2D compartido: la nube dañada conserva la estructura global del espécimen, lo que confirma que el daño es plausible y no ruido aleatorio. La perspectiva 2D muestra con precisión qué se perdió y qué se conservó. Ambas figuras se exportan listas para un artículo.

---

## Lámina 14 · Qué deja PaleoVox GUI a la paleontología (45 s)

En resumen: cuatro daños combinables, cinco pestañas, 34 controles documentados, cuatro formas de visualizar —vóxeles, malla PBR, t-SNE y 2D— y todo local, gratuito y de código abierto. Sirve a investigación, curaduría, museos y docencia. Y quiero ser honesto con los límites: es aumento sintético; aproxima la tafonomía, no reconstruye la verdad biológica del fósil, y los modelos entrenados con estos datos deben validarse. El siguiente paso es la generación en lote, semillas reproducibles y un adaptador para PyTorch.

---

## Lámina 15 · Gracias (20 s)

En conclusión: pasamos de un modelo imposible de conseguir a miles de variantes listas para entrenar, sin escribir una línea de código. PaleoVox GUI complementa el ecosistema PaleoVox y acerca la visualización avanzada a quien conoce el fósil, no la terminal. Gracias. ¿Preguntas?

---

## Notas de apoyo

- **Si preguntan por el repositorio:** github.com/ChuchoDC/PaleoVox (código abierto, MIT).
- **Si preguntan «¿esto reemplaza al fósil real?»:** no; es aumento de datos para entrenamiento, no sustituye curación ni análisis filogenético.
- **Si preguntan por otras herramientas (Blender, MeshLab):** existen; PaleoVox GUI es específico para daño tafonómico + flujo de aumento de datos reproducible en una sola interfaz.
- **Cifras de la demo:** cráneo con 1,530,891 vértices y 3,000,000 de triángulos; rejilla 128³ con 91,203 vóxeles tras el pipeline completo.
- **Tiempo de práctica sugerido:** dos ensayos cronometrados; si se excede, recortar las láminas 9 y 10 a 20 s cada una.
