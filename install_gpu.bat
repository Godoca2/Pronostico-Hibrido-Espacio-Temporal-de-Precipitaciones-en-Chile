@echo off
echo ========================================
echo Instalando TensorFlow GPU 2.10.0
echo ========================================
echo.
echo IMPORTANTE: Cierra VS Code antes de ejecutar
echo.
pause

call conda activate capstone
pip uninstall -y tensorflow tensorflow-gpu
pip install tensorflow-gpu==2.10.0

echo.
echo ========================================
echo Verificando instalacion GPU...
echo ========================================
python -c "import tensorflow as tf; print('\nTensorFlow version:', tf.__version__); gpus = tf.config.list_physical_devices('GPU'); print('GPUs detectadas:', len(gpus)); [print(f'  - {g.name}') for g in gpus]"

pause
