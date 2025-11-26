#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "png.h"
#include <vector>
#include <assert.h>
#include <iostream>
#include <memory>
#include "utils/image.h"
#include "utils/dct.h"
#include <string>
#include <chrono>
#include "mpi.h"

Image<float> get_srm_3x3() {
    std::cout << "Creating SRM 3x3 kernel..." << std::endl;
    Image<float> kernel(3, 3, 1);
    kernel.set(0, 0, 0, -1); kernel.set(0, 1, 0, 2); kernel.set(0, 2, 0, -1);
    kernel.set(1, 0, 0, 2); kernel.set(1, 1, 0, -4); kernel.set(1, 2, 0, 2);
    kernel.set(2, 0, 0, -1); kernel.set(2, 1, 0, 2); kernel.set(2, 2, 0, -1);
    return kernel;
}

Image<float> get_srm_5x5() {
    std::cout << "Creating SRM 5x5 kernel..." << std::endl;
    Image<float> kernel(5, 5, 1);
    kernel.set(0, 0, 0, -1); kernel.set(0, 1, 0, 2); kernel.set(0, 2, 0, -2); kernel.set(0, 3, 0, 2); kernel.set(0, 4, 0, -1);
    kernel.set(1, 0, 0, 2); kernel.set(1, 1, 0, -6); kernel.set(1, 2, 0, 8); kernel.set(1, 3, 0, -6); kernel.set(1, 4, 0, 2);
    kernel.set(2, 0, 0, -2); kernel.set(2, 1, 0, 8); kernel.set(2, 2, 0, -12); kernel.set(2, 3, 0, 8); kernel.set(2, 4, 0, -2);
    kernel.set(3, 0, 0, 2); kernel.set(3, 1, 0, -6); kernel.set(3, 2, 0, 8); kernel.set(3, 3, 0, -6); kernel.set(3, 4, 0, 2);
    kernel.set(4, 0, 0, -1); kernel.set(4, 1, 0, 2); kernel.set(4, 2, 0, -2); kernel.set(4, 3, 0, 2); kernel.set(4, 4, 0, -1);
    return kernel;
}

Image<float> get_srm_kernel(int size) {
    std::cout << "Getting SRM kernel of size " << size << "x" << size << std::endl;
    assert(size == 3 || size == 5);
    switch(size){
        case 3:
            return get_srm_3x3();
        case 5:
            return get_srm_5x5();
    }
    return get_srm_3x3();
}

Image<unsigned char> combine_blocks_to_image(const std::vector<Block<float>>& blocks, int width, int height) {
    std::cout << "Combining blocks into image of size " << width << "x" << height << "..." << std::endl;
    Image<unsigned char> result(width, height, 1);  // Crear una nueva imagen para el resultado

    for (const auto& block : blocks) {
        for (int row = 0; row < block.size; ++row) {
            for (int col = 0; col < block.size; ++col) {
                result.set(block.j + row, block.i + col, 0, static_cast<unsigned char>(block.get_pixel(row, col, 0)));
            }
        }
    }

    return result;
}

Image<unsigned char> compute_srm(const Image<unsigned char>& image, int kernel_size, int rank, int num_procs) {
    auto begin = std::chrono::steady_clock::now();
    std::cout << "Computing SRM " << kernel_size << "x" << kernel_size << " on process " << rank << "..." << std::endl;

    Image<float> srm = image.to_grayscale().convert<float>();
    Image<float> kernel = get_srm_kernel(kernel_size);  // Obtener el kernel SRM
    std::cout << "SRM and kernel loaded." << std::endl;

    std::vector<Block<float>> blocks = srm.get_blocks(8);  // Usar bloques de 8x8
    int num_blocks = blocks.size();
    std::cout << "Number of blocks: " << num_blocks << std::endl;

    int blocks_per_proc = num_blocks / num_procs;
    int start_block = rank * blocks_per_proc;
    int end_block = (rank == num_procs - 1) ? num_blocks : (rank + 1) * blocks_per_proc;

    std::cout << "Process " << rank << " processing blocks from " << start_block << " to " << end_block - 1 << std::endl;

    for (int i = start_block; i < end_block; ++i) {
        Block<float>& block = blocks[i];
        std::cout << "Process " << rank << " processing block " << i << " at (" << block.i << ", " << block.j << ")" << std::endl;

        int block_x = block.i;  // Coordenada x de la esquina superior izquierda del bloque
        int block_y = block.j;  // Coordenada y de la esquina superior izquierda del bloque
        Image<float> block_image(8, 8, 1);  // Crear una subimagen de 8x8
        for (int j = 0; j < block.size; ++j) {
            for (int i = 0; i < block.size; ++i) {
                block_image.set(j, i, 0, srm.get(block_y + j, block_x + i, 0));  // Copiar el pixel correspondiente
            }
        }

        Image<float> convolved_block = block_image.convolution(kernel);  // Aplicar convolución al bloque
        std::cout << "Process " << rank << " completed convolution on block " << i << std::endl;

        for (int j = 0; j < block.size; ++j) {
            for (int i = 0; i < block.size; ++i) {
                block.set_pixel(j, i, 0, convolved_block.get(j, i, 0));  // Reemplazar el bloque en la imagen
            }
        }
    }

    std::vector<Block<float>> all_blocks;
    if (rank == 0) {
        all_blocks.resize(num_blocks);
        std::cout << "Process 0 gathering all blocks..." << std::endl;
    }

    MPI_Gather(&blocks[start_block], blocks_per_proc * sizeof(Block<float>), MPI_BYTE, all_blocks.data(), blocks_per_proc * sizeof(Block<float>), MPI_BYTE, 0, MPI_COMM_WORLD);

    Image<unsigned char> result;
    if (rank == 0) {
        result = combine_blocks_to_image(all_blocks, srm.width, srm.height);
        std::cout << "Process 0 finished combining blocks into image." << std::endl;
    }

    auto end = std::chrono::steady_clock::now();
    if (rank == 0)
        std::cout << "SRM elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

    return result;
}

Image<unsigned char> compute_dct(const Image<unsigned char>& image, int block_size, bool invert, int rank, int num_procs) {
    auto begin = std::chrono::steady_clock::now();
    std::cout << "Computing DCT on process " << rank << "..." << std::endl;

    Image<float> grayscale = image.convert<float>().to_grayscale();
    std::vector<Block<float>> blocks = grayscale.get_blocks(block_size);  // Obtener bloques

    int num_blocks = blocks.size();
    std::cout << "Number of DCT blocks: " << num_blocks << std::endl;

    int blocks_per_proc = num_blocks / num_procs;
    int start_block = rank * blocks_per_proc;
    int end_block = (rank == num_procs - 1) ? num_blocks : (rank + 1) * blocks_per_proc;

    for (int i = start_block; i < end_block; ++i) {
        Block<float>& block = blocks[i];
        std::cout << "Process " << rank << " processing DCT block " << i << std::endl;

        float **dctBlock = dct::create_matrix(block_size, block_size);
        dct::direct(dctBlock, block, 0);

        if (invert) {
            std::cout << "Process " << rank << " applying inverse DCT on block " << i << std::endl;
            for (int k = 0; k < block.size / 2; ++k) {
                for (int l = 0; l < block.size / 2; ++l) {
                    dctBlock[k][l] = 0.0;  // Eliminar altas frecuencias
                }
            }
            dct::inverse(block, dctBlock, 0, 0.0, 255.);
        } else {
            dct::assign(dctBlock, block, 0);
        }

        dct::delete_matrix(dctBlock);
    }

    std::vector<Block<float>> all_blocks;
    if (rank == 0) {
        all_blocks.resize(num_blocks);  // Crear el búfer de recepción
    }

    MPI_Gather(&blocks[start_block], blocks_per_proc * sizeof(Block<float>), MPI_BYTE,
               all_blocks.data(), blocks_per_proc * sizeof(Block<float>), MPI_BYTE, 0, MPI_COMM_WORLD);

    Image<unsigned char> result;
    if (rank == 0) {
        result = combine_blocks_to_image(all_blocks, grayscale.width, grayscale.height);
        std::cout << "Process 0 finished DCT computation." << std::endl;
    }

    auto end = std::chrono::steady_clock::now();
    if (rank == 0)
        std::cout << "DCT elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

    return result;
}

Image<unsigned char> compute_ela(const Image<unsigned char>& image, int quality) {
    std::cout << "Computing ELA..." << std::endl;
    auto begin = std::chrono::steady_clock::now();
    
    Image<unsigned char> grayscale = image.to_grayscale();
    save_to_file("_temp.jpg", grayscale, quality);
    Image<float> compressed = load_from_file("_temp.jpg").convert<float>();
    compressed = compressed + (grayscale.convert<float>() * (-1));
    compressed = compressed.abs().normalized() * 255;
    Image<unsigned char> result = compressed.convert<unsigned char>();

    auto end = std::chrono::steady_clock::now();
    std::cout << "ELA elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

    return result;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, procs;
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::cout << "This is process " << rank << " of " << procs << std::endl;

    if (argc == 1) {
        std::cerr << "Image filename missing from arguments. Usage ./dct <filename>" << std::endl;
        exit(1);
    }

    Image<unsigned char> image = load_from_file(argv[1]);
    int block_size = 8;

    Image<unsigned char> srm3x3 = compute_srm(image, 3, rank, procs);
    save_to_file("srm_kernel_3x3.png", srm3x3);
    save_to_file("srm_kernel_5x5.png", compute_srm(image, 5, rank, procs));
    save_to_file("ela.png", compute_ela(image, 90)); // Esta parte no está paralelizada por simplicidad
    save_to_file("dct_invert.png", compute_dct(image, block_size, true, rank, procs));
    save_to_file("dct_direct.png", compute_dct(image, block_size, false, rank, procs));

    MPI_Finalize();
    return 0;
}
