from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi import HTTPException
import subprocess
import yaml
import os
import shutil
import logging
import uuid

app = FastAPI()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Extensiones de imagen a buscar
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/enhanceX2")
async def enhance_image(file: UploadFile = File(...)):
    try:
        # 1. Crear un directorio permanente para esta sesión dentro de "enhanced_outputs"
        session_id = uuid.uuid4().hex
        base_output_dir = os.path.join(os.getcwd(), "enhanced_outputs", session_id)
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Crear subdirectorio para la imagen de entrada
        input_images_dir = os.path.join(base_output_dir, "input_images")
        os.makedirs(input_images_dir, exist_ok=True)
        input_path = os.path.join(input_images_dir, "input.png")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Cargar el YAML y actualizar parámetros
        config_path = "options/test/HMA_SRx2.yml"
        with open(config_path, "r") as f:
            opt = yaml.safe_load(f)
        
        # Deshabilitar métricas para evitar errores (PSNR/SSIM)
        if 'val' in opt and 'metrics' in opt['val']:
            opt['val']['metrics'] = {}
        
        # Forzar un sufijo para la imagen de salida si no existe
        if not opt['val'].get('suffix'):
            opt['val']['suffix'] = "enhanced"
        
        # Actualizar la configuración del dataset para que use el directorio de entrada
        opt['datasets'] = {
            'test_1': {
                'type': 'InferenceDataset',
                'name': 'Inference',  # Este valor se usará para crear la carpeta de salida
                'dataroot_lq': input_images_dir,
                'filename_tmpl': '{}',
                'phase': 'test',
                'scale': 2,
                'io_backend': {'type': 'disk'}
            }
        }
        
        # Opcional: definir result_path (puede ser ignorado por el script)
        opt['result_path'] = os.path.join(base_output_dir, "output.png")
        
        # Forzar que las rutas de salida queden en el directorio permanente
        if 'path' not in opt:
            opt['path'] = {}
        opt['path']['results_root'] = base_output_dir
        # Se define que las imágenes se guarden en "visualization" dentro del directorio permanente
        opt['path']['visualization'] = os.path.join(base_output_dir, "visualization")
        os.makedirs(opt['path']['visualization'], exist_ok=True)
        # Crear la carpeta para el dataset de validación (en este caso "Inference")
        dataset_vis_dir = os.path.join(opt['path']['visualization'], "Inference")
        os.makedirs(dataset_vis_dir, exist_ok=True)
        
        # Ruta del modelo preentrenado (asegúrate de que sea correcta en tu entorno)
        opt['path']['pretrain_network_g'] = "./experiments/pretrained_models/HMA_SRx2_pretrain.pth"
        
        # 3. Guardar la configuración modificada en un YAML permanente
        permanent_yaml = os.path.join(base_output_dir, "config.yml")
        with open(permanent_yaml, "w") as f:
            yaml.dump(opt, f)
        
        # 4. Ejecutar el modelo (inferencia)
        command = ["python3", "modelo/test.py", "-opt", permanent_yaml]
        subprocess.run(command, check=True)
        
        # 3. Definir la ruta de la imagen mejorada
        output_path = "results/HMA-X2/visualization/Inference/input_enhanced.png"

        # 4. Verificar si la imagen mejorada existe
        if not os.path.exists(output_path):
            raise HTTPException(status_code=404, detail="Imagen mejorada no encontrada")

        # 5. Devolver la imagen mejorada
        return FileResponse(output_path, media_type="image/png")

        # 5. Buscar la imagen de salida en el directorio de visualización del dataset ("Inference")
        dataset_vis_dir = os.path.join(opt['path']['visualization'], "Inference")
        logger.info(f"Buscando imágenes en: {dataset_vis_dir}")
        if os.path.isdir(dataset_vis_dir):
            files = os.listdir(dataset_vis_dir)
            logger.info(f"Archivos encontrados: {files}")
            output_file = None
            for f in files:
                name, ext = os.path.splitext(f)
                if ext.lower() in IMAGE_EXTENSIONS:
                    output_file = os.path.join(dataset_vis_dir, f)
                    logger.info(f"Archivo de salida encontrado: {output_file}")
                    break
        else:
            raise FileNotFoundError(f"No existe el directorio de visualización: {dataset_vis_dir}")
        
        if output_file is None:
            raise FileNotFoundError("No se encontró la imagen de salida generada.")
        
        return FileResponse(output_file, media_type="image/png")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return {"error": str(e)}


@app.post("/enhanceX3")
async def enhance_image(file: UploadFile = File(...)):
    try:
        # 1. Crear un directorio permanente para esta sesión dentro de "enhanced_outputs"
        session_id = uuid.uuid4().hex
        base_output_dir = os.path.join(os.getcwd(), "enhanced_outputs", session_id)
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Crear subdirectorio para la imagen de entrada
        input_images_dir = os.path.join(base_output_dir, "input_images")
        os.makedirs(input_images_dir, exist_ok=True)
        input_path = os.path.join(input_images_dir, "input.png")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Cargar el YAML y actualizar parámetros
        config_path = "options/test/HMA_SRx3.yml"
        with open(config_path, "r") as f:
            opt = yaml.safe_load(f)
        
        # Deshabilitar métricas para evitar errores (PSNR/SSIM)
        if 'val' in opt and 'metrics' in opt['val']:
            opt['val']['metrics'] = {}
        
        # Forzar un sufijo para la imagen de salida si no existe
        if not opt['val'].get('suffix'):
            opt['val']['suffix'] = "enhanced"
        
        # Actualizar la configuración del dataset para que use el directorio de entrada
        opt['datasets'] = {
            'test_1': {
                'type': 'InferenceDataset',
                'name': 'Inference',  # Este valor se usará para crear la carpeta de salida
                'dataroot_lq': input_images_dir,
                'filename_tmpl': '{}',
                'phase': 'test',
                'scale': 3,
                'io_backend': {'type': 'disk'}
            }
        }
        
        # Opcional: definir result_path (puede ser ignorado por el script)
        opt['result_path'] = os.path.join(base_output_dir, "output.png")
        
        # Forzar que las rutas de salida queden en el directorio permanente
        if 'path' not in opt:
            opt['path'] = {}
        opt['path']['results_root'] = base_output_dir
        # Se define que las imágenes se guarden en "visualization" dentro del directorio permanente
        opt['path']['visualization'] = os.path.join(base_output_dir, "visualization")
        os.makedirs(opt['path']['visualization'], exist_ok=True)
        # Crear la carpeta para el dataset de validación (en este caso "Inference")
        dataset_vis_dir = os.path.join(opt['path']['visualization'], "Inference")
        os.makedirs(dataset_vis_dir, exist_ok=True)
        
        # Ruta del modelo preentrenado (asegúrate de que sea correcta en tu entorno)
        opt['path']['pretrain_network_g'] = "./experiments/pretrained_models/HMA_SRx3_pretrain.pth"
        
        # 3. Guardar la configuración modificada en un YAML permanente
        permanent_yaml = os.path.join(base_output_dir, "config.yml")
        with open(permanent_yaml, "w") as f:
            yaml.dump(opt, f)
        
        # 4. Ejecutar el modelo (inferencia)
        command = ["python3", "modelo/test.py", "-opt", permanent_yaml]
        subprocess.run(command, check=True)
        
        # 3. Definir la ruta de la imagen mejorada
        output_path = "results/HMA-X3/visualization/Inference/input_enhanced.png"

        # 4. Verificar si la imagen mejorada existe
        if not os.path.exists(output_path):
            raise HTTPException(status_code=404, detail="Imagen mejorada no encontrada")

        # 5. Devolver la imagen mejorada
        return FileResponse(output_path, media_type="image/png")

        # 5. Buscar la imagen de salida en el directorio de visualización del dataset ("Inference")
        dataset_vis_dir = os.path.join(opt['path']['visualization'], "Inference")
        logger.info(f"Buscando imágenes en: {dataset_vis_dir}")
        if os.path.isdir(dataset_vis_dir):
            files = os.listdir(dataset_vis_dir)
            logger.info(f"Archivos encontrados: {files}")
            output_file = None
            for f in files:
                name, ext = os.path.splitext(f)
                if ext.lower() in IMAGE_EXTENSIONS:
                    output_file = os.path.join(dataset_vis_dir, f)
                    logger.info(f"Archivo de salida encontrado: {output_file}")
                    break
        else:
            raise FileNotFoundError(f"No existe el directorio de visualización: {dataset_vis_dir}")
        
        if output_file is None:
            raise FileNotFoundError("No se encontró la imagen de salida generada.")
        
        return FileResponse(output_file, media_type="image/png")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return {"error": str(e)}



@app.post("/enhanceX4")
async def enhance_image(file: UploadFile = File(...)):
    try:
        # 1. Crear un directorio permanente para esta sesión dentro de "enhanced_outputs"
        session_id = uuid.uuid4().hex
        base_output_dir = os.path.join(os.getcwd(), "enhanced_outputs", session_id)
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Crear subdirectorio para la imagen de entrada
        input_images_dir = os.path.join(base_output_dir, "input_images")
        os.makedirs(input_images_dir, exist_ok=True)
        input_path = os.path.join(input_images_dir, "input.png")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Cargar el YAML y actualizar parámetros
        config_path = "options/test/HMA_SRx4.yml"
        with open(config_path, "r") as f:
            opt = yaml.safe_load(f)
        
        # Deshabilitar métricas para evitar errores (PSNR/SSIM)
        if 'val' in opt and 'metrics' in opt['val']:
            opt['val']['metrics'] = {}
        
        # Forzar un sufijo para la imagen de salida si no existe
        if not opt['val'].get('suffix'):
            opt['val']['suffix'] = "enhanced"
        
        # Actualizar la configuración del dataset para que use el directorio de entrada
        opt['datasets'] = {
            'test_1': {
                'type': 'InferenceDataset',
                'name': 'Inference',  # Este valor se usará para crear la carpeta de salida
                'dataroot_lq': input_images_dir,
                'filename_tmpl': '{}',
                'phase': 'test',
                'scale': 4,
                'io_backend': {'type': 'disk'}
            }
        }
        
        # Opcional: definir result_path (puede ser ignorado por el script)
        opt['result_path'] = os.path.join(base_output_dir, "output.png")
        
        # Forzar que las rutas de salida queden en el directorio permanente
        if 'path' not in opt:
            opt['path'] = {}
        opt['path']['results_root'] = base_output_dir
        # Se define que las imágenes se guarden en "visualization" dentro del directorio permanente
        opt['path']['visualization'] = os.path.join(base_output_dir, "visualization")
        os.makedirs(opt['path']['visualization'], exist_ok=True)
        # Crear la carpeta para el dataset de validación (en este caso "Inference")
        dataset_vis_dir = os.path.join(opt['path']['visualization'], "Inference")
        os.makedirs(dataset_vis_dir, exist_ok=True)
        
        # Ruta del modelo preentrenado (asegúrate de que sea correcta en tu entorno)
        opt['path']['pretrain_network_g'] = "./experiments/pretrained_models/HMA_SRx4_pretrain.pth"
        
        # 3. Guardar la configuración modificada en un YAML permanente
        permanent_yaml = os.path.join(base_output_dir, "config.yml")
        with open(permanent_yaml, "w") as f:
            yaml.dump(opt, f)
        
        # 4. Ejecutar el modelo (inferencia)
        command = ["python3", "modelo/test.py", "-opt", permanent_yaml]
        subprocess.run(command, check=True)
        
        # 3. Definir la ruta de la imagen mejorada
        output_path = "results/HMA-X4/visualization/Inference/input_enhanced.png"

        # 4. Verificar si la imagen mejorada existe
        if not os.path.exists(output_path):
            raise HTTPException(status_code=404, detail="Imagen mejorada no encontrada")

        # 5. Devolver la imagen mejorada
        return FileResponse(output_path, media_type="image/png")

        # 5. Buscar la imagen de salida en el directorio de visualización del dataset ("Inference")
        dataset_vis_dir = os.path.join(opt['path']['visualization'], "Inference")
        logger.info(f"Buscando imágenes en: {dataset_vis_dir}")
        if os.path.isdir(dataset_vis_dir):
            files = os.listdir(dataset_vis_dir)
            logger.info(f"Archivos encontrados: {files}")
            output_file = None
            for f in files:
                name, ext = os.path.splitext(f)
                if ext.lower() in IMAGE_EXTENSIONS:
                    output_file = os.path.join(dataset_vis_dir, f)
                    logger.info(f"Archivo de salida encontrado: {output_file}")
                    break
        else:
            raise FileNotFoundError(f"No existe el directorio de visualización: {dataset_vis_dir}")
        
        if output_file is None:
            raise FileNotFoundError("No se encontró la imagen de salida generada.")
        
        return FileResponse(output_file, media_type="image/png")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return {"error": str(e)}

    
@app.post("/enhanceX4Real")
async def enhance_image(file: UploadFile = File(...)):
    try:
        # 1. Crear un directorio permanente para esta sesión dentro de "enhanced_outputs"
        session_id = uuid.uuid4().hex
        base_output_dir = os.path.join(os.getcwd(), "enhanced_outputs", session_id)
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Crear subdirectorio para la imagen de entrada
        input_images_dir = os.path.join(base_output_dir, "input_images")
        os.makedirs(input_images_dir, exist_ok=True)
        input_path = os.path.join(input_images_dir, "input.png")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Cargar el YAML y actualizar parámetros
        config_path = "options/test/HAT_GAN_Real_SRx4.yml"
        with open(config_path, "r") as f:
            opt = yaml.safe_load(f)
        
        # Deshabilitar métricas para evitar errores (PSNR/SSIM)
        if 'val' in opt and 'metrics' in opt['val']:
            opt['val']['metrics'] = {}
        
        # Forzar un sufijo para la imagen de salida si no existe
        if not opt['val'].get('suffix'):
            opt['val']['suffix'] = "enhanced"
        
        # Actualizar la configuración del dataset para que use el directorio de entrada
        opt['datasets'] = {
            'test_1': {
                'type': 'InferenceDataset',
                'name': 'Inference',  # Este valor se usará para crear la carpeta de salida
                'dataroot_lq': input_images_dir,
                'filename_tmpl': '{}',
                'phase': 'test',
                'scale': 4,
                'io_backend': {'type': 'disk'}
            }
        }
        
        # Opcional: definir result_path (puede ser ignorado por el script)
        opt['result_path'] = os.path.join(base_output_dir, "output.png")
        
        # Forzar que las rutas de salida queden en el directorio permanente
        if 'path' not in opt:
            opt['path'] = {}
        opt['path']['results_root'] = base_output_dir
        # Se define que las imágenes se guarden en "visualization" dentro del directorio permanente
        opt['path']['visualization'] = os.path.join(base_output_dir, "visualization")
        os.makedirs(opt['path']['visualization'], exist_ok=True)
        # Crear la carpeta para el dataset de validación (en este caso "Inference")
        dataset_vis_dir = os.path.join(opt['path']['visualization'], "Inference")
        os.makedirs(dataset_vis_dir, exist_ok=True)
        
        # Ruta del modelo preentrenado (asegúrate de que sea correcta en tu entorno)
        opt['path']['pretrain_network_g'] = "./experiments/pretrained_models/Real_HAT_GAN_sharper.pth"
        
        # 3. Guardar la configuración modificada en un YAML permanente
        permanent_yaml = os.path.join(base_output_dir, "config.yml")
        with open(permanent_yaml, "w") as f:
            yaml.dump(opt, f)
        
        # 4. Ejecutar el modelo (inferencia)
        command = ["python3", "modelo/test.py", "-opt", permanent_yaml]
        subprocess.run(command, check=True)
        
        # 3. Definir la ruta de la imagen mejorada
        output_path = "results/HAT_GAN_Real_sharper/visualization/Inference/input_enhanced.png"

        # 4. Verificar si la imagen mejorada existe
        if not os.path.exists(output_path):
            raise HTTPException(status_code=404, detail="Imagen mejorada no encontrada")

        # 5. Devolver la imagen mejorada
        return FileResponse(output_path, media_type="image/png")

        # 5. Buscar la imagen de salida en el directorio de visualización del dataset ("Inference")
        dataset_vis_dir = os.path.join(opt['path']['visualization'], "Inference")
        logger.info(f"Buscando imágenes en: {dataset_vis_dir}")
        if os.path.isdir(dataset_vis_dir):
            files = os.listdir(dataset_vis_dir)
            logger.info(f"Archivos encontrados: {files}")
            output_file = None
            for f in files:
                name, ext = os.path.splitext(f)
                if ext.lower() in IMAGE_EXTENSIONS:
                    output_file = os.path.join(dataset_vis_dir, f)
                    logger.info(f"Archivo de salida encontrado: {output_file}")
                    break
        else:
            raise FileNotFoundError(f"No existe el directorio de visualización: {dataset_vis_dir}")
        
        if output_file is None:
            raise FileNotFoundError("No se encontró la imagen de salida generada.")
        
        return FileResponse(output_file, media_type="image/png")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return {"error": str(e)}