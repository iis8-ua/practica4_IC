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
    Image<float> kernel(3, 3, 1);
    kernel.set(0, 0, 0, -1); kernel.set(0, 1, 0, 2); kernel.set(0, 2, 0, -1);
    kernel.set(1, 0, 0, 2); kernel.set(1, 1, 0, -4); kernel.set(1, 2, 0, 2);
    kernel.set(2, 0, 0, -1); kernel.set(2, 1, 0, 2); kernel.set(2, 2, 0, -1);
    return kernel;
}

Image<float> get_srm_5x5() {
    Image<float> kernel(5, 5, 1);
    kernel.set(0, 0, 0, -1); kernel.set(0, 1, 0, 2); kernel.set(0, 2, 0, -2); kernel.set(0, 3, 0, 2); kernel.set(0, 4, 0, -1);
    kernel.set(1, 0, 0, 2); kernel.set(1, 1, 0, -6); kernel.set(1, 2, 0, 8); kernel.set(1, 3, 0, -6); kernel.set(1, 4, 0, 2);
    kernel.set(2, 0, 0, -2); kernel.set(2, 1, 0, 8); kernel.set(2, 2, 0, -12); kernel.set(2, 3, 0, 8); kernel.set(2, 4, 0, -2);
    kernel.set(3, 0, 0, 2); kernel.set(3, 1, 0, -6); kernel.set(3, 2, 0, 8); kernel.set(3, 3, 0, -6); kernel.set(3, 4, 0, 2);
    kernel.set(4, 0, 0, -1); kernel.set(4, 1, 0, 2); kernel.set(4, 2, 0, -2); kernel.set(4, 3, 0, 2); kernel.set(4, 4, 0, -1);
    return kernel;
}

Image<float> get_srm_kernel(int size) {
    assert(size == 3 || size == 5);
    switch(size){
        case 3:
            return get_srm_3x3();
        case 5:
            return get_srm_5x5();
    }
    return get_srm_3x3();
}

static void compute_block_range(int num_blocks, int rank, int num_procs, int &start, int &end) {
    int base = num_blocks / num_procs;
    int extra = num_blocks % num_procs;
    if (rank < extra) {
        start = rank * (base + 1);
        end   = start + (base + 1);
    } else {
        start = extra * (base + 1) + (rank - extra) * base;
        end   = start + base;
    }
    if (start > num_blocks) start = num_blocks;
    if (end > num_blocks)   end   = num_blocks;
}

Image<unsigned char> compute_srm(const Image<unsigned char> &image, int kernel_size, int rank, int procs) {
    auto begin = std::chrono::steady_clock::now();
    if (rank == 0)
        std::cout<<"Computing SRM "<<kernel_size<<"x"<<kernel_size<<"..."<<std::endl;

    int width = 0, height = 0;
    if (rank == 0) {
        width  = image.width;
        height = image.height;
    }
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    Image<unsigned char> local_img;
    if (rank == 0) local_img = image;
    else           local_img = Image<unsigned char>(width, height, image.channels);

    local_img.broadcast(0);

    Image<float> srm = local_img.to_grayscale().convert<float>();
    Image<float> kernel = get_srm_kernel(kernel_size);

    std::vector<Block<float>> blocks = srm.get_blocks(8);
    int num_blocks = (int)blocks.size();

    int start, end;
    compute_block_range(num_blocks, rank, procs, start, end);

    for (int idx = start; idx < end; ++idx) {
        Block<float> &block = blocks[idx];
        Image<float> block_img(block.size, block.size, 1);
        for (int j = 0; j < block.size; ++j)
            for (int i = 0; i < block.size; ++i)
                block_img.set(j, i, 0, srm.get(block.j + j, block.i + i, 0));

        Image<float> convolved = block_img.convolution(kernel);
        for (int j = 0; j < block.size; ++j)
            for (int i = 0; i < block.size; ++i)
                block.set_pixel(j, i, 0, convolved.get(j, i, 0));
    }

    for (int idx = 0; idx < num_blocks; ++idx) {
        Block<float> &block = blocks[idx];
        for (int j = 0; j < block.size; ++j)
            for (int i = 0; i < block.size; ++i) {
                float v = block.get_pixel(j, i, 0);
                srm.set(block.j + j, block.i + i, 0, v);
            }
    }

    Image<float> srm_abs = srm.abs().normalized() * 255.0f;
    Image<unsigned char> local_result = srm_abs.convert<unsigned char>();

    Image<unsigned char> result;
    if (rank == 0) result = local_result;
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        auto end_t = std::chrono::steady_clock::now();
        std::cout<<"SRM elapsed time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t - begin).count()<<"ms"<<std::endl;
    }
    return result;
}

Image<unsigned char> compute_dct(const Image<unsigned char> &image, int block_size, bool invert, int rank, int procs) {
    auto begin = std::chrono::steady_clock::now();
    if (rank == 0) {
        std::cout<<"Computing";
        if (invert) std::cout<<" inverse";
        else std::cout<<" direct";
        std::cout<<" DCT "<<block_size<<"x"<<block_size<<"..."<<std::endl;
    }

    int width = 0, height = 0;
    if (rank == 0) {
        width  = image.width;
        height = image.height;
    }
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    Image<unsigned char> local_img;
    if (rank == 0) local_img = image;
    else           local_img = Image<unsigned char>(width, height, image.channels);
    local_img.broadcast(0);

    Image<float> grayscale = local_img.convert<float>().to_grayscale();
    std::vector<Block<float>> blocks = grayscale.get_blocks(block_size);
    int num_blocks = (int)blocks.size();

    int start, end;
    compute_block_range(num_blocks, rank, procs, start, end);

    for (int i = start; i < end; ++i) {
        Block<float> &block = blocks[i];
        float **dctBlock = dct::create_matrix(block_size, block_size);
        dct::direct(dctBlock, block, 0);
        if (invert) {
            for (int k = 0; k < block.size / 2; ++k)
                for (int l = 0; l < block.size / 2; ++l)
                    dctBlock[k][l] = 0.0;
            dct::inverse(block, dctBlock, 0, 0.0, 255.);
        } else {
            dct::assign(dctBlock, block, 0);
        }
        dct::delete_matrix(dctBlock);
    }

    for (int i = 0; i < num_blocks; ++i) {
        Block<float> &block = blocks[i];
        for (int j = 0; j < block.size; ++j)
            for (int k = 0; k < block.size; ++k)
                grayscale.set(block.j + j, block.i + k, 0, block.get_pixel(j, k, 0));
    }

    Image<unsigned char> local_result = grayscale.convert<unsigned char>();
    Image<unsigned char> result;
    if (rank == 0) result = local_result;
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        auto end_t = std::chrono::steady_clock::now();
        std::cout<<"DCT elapsed time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t - begin).count()<<"ms"<<std::endl;
    }
    return result;
}

Image<unsigned char> compute_ela(const Image<unsigned char> &image, int quality){
    std::cout<<"Computing ELA..."<<std::endl;
    auto begin = std::chrono::steady_clock::now();
    Image<unsigned char> grayscale = image.to_grayscale();
    save_to_file("_temp.jpg", grayscale, quality);
    Image<float> compressed = load_from_file("_temp.jpg").convert<float>();
    compressed = compressed + (grayscale.convert<float>()*(-1));
    compressed = compressed.abs().normalized() * 255;
    Image<unsigned char> result = compressed.convert<unsigned char>();
    auto end = std::chrono::steady_clock::now();
    std::cout<<"ELA elapsed time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"ms"<<std::endl;
    return result;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, procs;
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::cout<<"This is process "<<rank<<" of "<<procs<<std::endl;
    if(argc == 1) {
        if (rank == 0)
            std::cerr<<"Image filename missing from arguments. Usage ./dct <filename>"<<std::endl;
        MPI_Finalize();
        return 1;
    }

    int block_size=8;
    Image<unsigned char> image;

    if (rank == 0)
        image = load_from_file(argv[1]);

    int width = 0, height = 0, channels = 0;
    if (rank == 0) {
        width = image.width;
        height = image.height;
        channels = image.channels;
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
        image = Image<unsigned char>(width, height, channels);

    image.broadcast(0);

    Image<unsigned char> srm3x3 = compute_srm(image, 3, rank, procs);
    if (rank == 0) {
        save_to_file("srm_kernel_3x3.png", srm3x3);
        save_to_file("srm_kernel_5x5.png", compute_srm(image, 5, rank, procs));
        save_to_file("ela.png", compute_ela(image, 90));
        save_to_file("dct_invert.png", compute_dct(image, block_size, true, rank, procs));
        save_to_file("dct_direct.png", compute_dct(image, block_size, false, rank, procs));
    }

    MPI_Finalize();
    return 0;
}
