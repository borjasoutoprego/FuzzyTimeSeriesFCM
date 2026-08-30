# Instrucciones para Codex

## Contexto

Este repositorio corresponde a un Trabajo Fin de Máster (TFM) relacionado con el análisis y predicción de series temporales basado en cluster fuzzy.

El código está desarrollado principalmente en Python.

El objetivo es desarrollar, experimentar, evaluar y comparar diferentes técnicas de análisis y predicción de series temporales basado en cluster fuzzy.

## Regla principal

Antes de realizar cambios importantes, comprende primero la estructura existente del proyecto y el código relacionado.

No rehagas partes del proyecto que ya funcionan correctamente salvo que el usuario lo solicite explícitamente.

Prioriza cambios pequeños, claros y fáciles de revisar.

## Modificaciones

Cuando se solicite una modificación:

1. Inspecciona primero el código relevante.
2. Comprueba cómo encaja el cambio con la estructura existente.
3. Realiza únicamente los cambios necesarios.
4. No modifiques archivos no relacionados con la tarea.
5. Evita introducir dependencias nuevas salvo que sean necesarias.
6. No cambies la metodología estadística o los modelos utilizados sin indicación explícita del usuario.
7. No elimines código o funcionalidades existentes simplemente porque parezcan innecesarias.

## Datos

Los datos originales son de solo lectura.

Nunca sobrescribas, elimines o alteres los datos originales salvo que el usuario lo solicite explícitamente.

Si necesitas transformar datos, genera un nuevo archivo o realiza la transformación mediante código.

No subas al repositorio datos privados, credenciales, claves, tokens o información sensible.

## Python

Utiliza Python siguiendo un estilo claro y sencillo.

Prioriza las librerías ya utilizadas en el proyecto.

Antes de añadir una librería nueva, comprueba si la funcionalidad puede realizarse con las dependencias existentes.

No cambies versiones de dependencias sin una razón clara.

## Análisis estadístico

Los cambios relacionados con modelos estadísticos o de aprendizaje automático deben mantener la coherencia con la metodología del TFM.

No inventes resultados.

No afirmes que un modelo funciona mejor que otro sin ejecutar el análisis correspondiente.

Distingue siempre entre:

- resultados obtenidos mediante ejecución;
- hipótesis o propuestas;
- interpretaciones.

Cuando un resultado sea relevante, indica cómo se ha obtenido.

## Resultados

No modifiques manualmente resultados experimentales para mejorar su apariencia.

Los gráficos, tablas y métricas deben proceder de los datos y ejecuciones reales.

Si un resultado parece incorrecto, investiga primero la causa antes de modificar el código para obtener un resultado esperado.

## Ejecución y pruebas

Después de realizar cambios importantes:

1. Ejecuta las pruebas existentes.
2. Si no existen pruebas, ejecuta al menos el código afectado cuando sea posible.
3. Comprueba que no se han introducido errores.
4. Revisa `git diff` antes de considerar terminado el trabajo.

No ejecutes comandos destructivos.

## Git

Trabaja preferentemente en una rama distinta de `main`.

No hagas `push` directamente a `main` salvo que el usuario lo solicite explícitamente.

Antes de hacer commit:

```bash
git status
git diff
```

Revisa los cambios realizados.

Los commits deben ser pequeños y descriptivos.

Ejemplos:

```text
Add time series preprocessing
Fix missing date handling
Add ARIMA evaluation
```

## Pull Requests

Cuando el usuario solicite crear un Pull Request:

1. Comprueba el estado del repositorio.
2. Comprueba los cambios incluidos.
3. Ejecuta las pruebas relevantes.
4. Haz push de la rama.
5. Crea el Pull Request hacia la rama principal.
6. Describe brevemente:
   - qué se ha cambiado;
   - por qué;
   - cómo se ha probado.

No hagas merge del Pull Request salvo que el usuario lo solicite explícitamente.

## Seguridad

Nunca:

- muestres credenciales;
- copies tokens a archivos del proyecto;
- subas claves SSH;
- subas `.env` con secretos;
- ejecutes comandos destructivos sin autorización;
- elimines repositorios;
- modifiques configuraciones de GitHub sin solicitarlo.

Si encuentras una credencial o secreto en el repositorio, detente y avisa al usuario.

## Comunicación

Antes de realizar una operación potencialmente destructiva o difícil de revertir, solicita confirmación.

Para cambios normales de código puedes proceder directamente si el usuario ha dado una instrucción clara.

Al finalizar una tarea, resume:

1. Qué has cambiado.
2. Qué archivos has modificado.
3. Qué pruebas has ejecutado.
4. Si existe algún problema pendiente.

No inventes pruebas ni resultados que no hayas ejecutado.
