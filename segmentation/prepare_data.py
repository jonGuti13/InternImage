import os
import glob
import argparse
import shutil

def deleteFolderContent(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        os.remove(file_path)

    return


def copyFolderContent(base_folder, fold_number, preprocessing_code, exp_number, img_or_msk, destination_folder):
    if img_or_msk == 'img':
        source_folder = os.path.join(base_folder, "Fold" + str(fold_number), "Cube_" + preprocessing_code + '_Npy/')
        file_pattern = "nf*.npy"
    elif img_or_msk == 'msk':
        source_folder = os.path.join(base_folder, "Fold" + str(fold_number), "Labels_" + preprocessing_code[0:7] + '/Exp' + exp_number + '/')
        file_pattern = "nf*.png"

    matching_files = glob.glob(f"{source_folder}{file_pattern}")

    # Loop through the matching files and copy them to the destination folder
    for source_file in matching_files:
    # Get the file name without the path
        file_name = source_file.split("/")[-1]

        # Construct the destination path
        destination_file = f"{destination_folder}{file_name}"

        # Copy the file to the destination
        shutil.copy(source_file, destination_file)

        print(f"File '{source_file}' copied to '{destination_file}'")

    return


def changeFileNames(npy_folder, png_folder):
    # Obtener la lista de archivos en ambas carpetas
    archivos_npy = os.listdir(npy_folder)
    archivos_png = os.listdir(png_folder)

    archivos_npy.sort()
    archivos_png.sort()

    print(len(archivos_png))

    # Iterar a través de los archivos en la carpeta NPY
    for i in range(len(archivos_npy)):
        archivo_npy = archivos_npy[i]

        # Obtener el nombre base (sin la extensión) del archivo NPY
        nombre_base = os.path.splitext(archivo_npy)[0]
        # Crear el nuevo nombre del archivo en formato nfXXXX_YYY_TC_PN.png
        nuevo_nombre = f"{nombre_base}.png"

        # Ruta completa de origen y destino
        origen = os.path.join(png_folder, archivos_png[i])
        destino = os.path.join(png_folder, nuevo_nombre)

        # Renombrar el archivo
        os.rename(origen, destino)
        print(f"Renombrado: {archivos_png[i]} => {nuevo_nombre}")

    return


def main():
    ap = argparse.ArgumentParser(description='Change name of annotations to have same name as images')
    ap.add_argument('--base_folder', default='/work/jon/vault/phd_data/HSI-Drive_2.0_Jon')
    ap.add_argument("--training_folds", nargs="+", type=int, required=True)
    ap.add_argument("--validation_folds", nargs="+", type=int, required=True)
    ap.add_argument("--test_folds", nargs="+", type=int, required=True)
    ap.add_argument('--exp_number', type=str, required=True)
    ap.add_argument('--preprocessing_code', default='208_400_TC_PN')
    ap.add_argument('--npy_folder', help='path of the npy folder where images are stored', required=True)
    ap.add_argument('--png_folder', help='path of the png folder where annotations are stored', required=True)

    args = ap.parse_args()

    deleteFolderContent(args.npy_folder + 'training/')
    deleteFolderContent(args.png_folder + 'training/')
    deleteFolderContent(args.npy_folder + 'validation/')
    deleteFolderContent(args.png_folder + 'validation/')
    deleteFolderContent(args.npy_folder + 'test/')
    deleteFolderContent(args.png_folder + 'test/')


    for i, fold_number in enumerate(args.training_folds):
        copyFolderContent(args.base_folder, fold_number, args.preprocessing_code, args.exp_number, 'img', args.npy_folder + 'training/')
        copyFolderContent(args.base_folder, fold_number, args.preprocessing_code, args.exp_number, 'msk', args.png_folder + 'training/')

    for i, fold_number in enumerate(args.validation_folds):
        copyFolderContent(args.base_folder, fold_number, args.preprocessing_code, args.exp_number, 'img', args.npy_folder + 'validation/')
        copyFolderContent(args.base_folder, fold_number, args.preprocessing_code, args.exp_number, 'msk', args.png_folder + 'validation/')

    for i, fold_number in enumerate(args.test_folds):
        copyFolderContent(args.base_folder, fold_number, args.preprocessing_code, args.exp_number, 'img', args.npy_folder + 'test/')
        copyFolderContent(args.base_folder, fold_number, args.preprocessing_code, args.exp_number, 'msk', args.png_folder + 'test/')


    changeFileNames(args.npy_folder + 'training/', args.png_folder + 'training/')
    changeFileNames(args.npy_folder + 'validation/', args.png_folder + 'validation/')
    changeFileNames(args.npy_folder + 'test/', args.png_folder + 'test/')

    #python3 prepare_data.py --training_folds 1 2 3 --validation_folds 4 --npy_folder /work/jon/vault/phd_data/HSI-Drive_2.0_Jon/openmm/images/ --png_folder /work/jon/vault/phd_data/HSI-Drive_2.0_Jon/openmm/annotations/ --exp_number 104


if __name__ == '__main__':
    main()