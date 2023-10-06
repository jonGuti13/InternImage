#Seguro que hay una forma utilizando scripts de openmm de generar automáticamente las imágenes de salida coloreadas, pero me es más fácil
#convertirlas a .png y luego visualizarlas y pintarlas con Matlab.

import pickle
import argparse
import cv2
import os

def main():
    ap = argparse.ArgumentParser(description='Change name of annotations to have same name as images')
    ap.add_argument('pkl_file')
    ap.add_argument('output_folder', default='./work_dirs/upernet_internimage_t_512x1024_160k_hsidrive/Fold0/test_2023_09_27/')
    ap.add_argument('reference_names_folder', default='/data/HSI-Drive/2.0/openmm/annotations/test')
    ap.add_argument('--output_fmt', default='png')
    args = ap.parse_args()

    with open(args.pkl_file, 'rb') as f:
        data = pickle.load(f)

    output_names = os.listdir(args.reference_names_folder)
    output_names.sort()

    for i, output in enumerate(data):
        cv2.imwrite(os.path.join(args.output_folder, output_names[i][:-3] + args.output_fmt), output)

    return

if __name__ == '__main__':
    main()