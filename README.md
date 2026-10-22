# CCMpred

[![build](https://github.com/pstansfeld/CCMpred/actions/workflows/build.yml/badge.svg)](https://github.com/pstansfeld/CCMpred/actions/workflows/build.yml)

Protein Residue-Residue **C**ontacts from **C**orrelated **M**utations **pred**icted quickly and accurately.

CCMpred is a C implementation of a Markov Random Field pseudo-likelihood maximization for learning protein residue-residue contacts as made popular by Ekeberg et al. [1] and Balakrishnan and Kamisetty [2]. While predicting contacts with comparable accuracy to the referenced methods, however, CCMpred is written in C / CUDA C, performance-tuned and therefore much faster.

## About this fork

This is a maintained fork of [soedinglab/CCMpred](https://github.com/soedinglab/CCMpred) (Seemayer, Gruber & Söding [3]). The method and the output are unchanged; the changes are to building and speed:

  * **OpenMP runs in parallel.** The CPU path (`-t`) previously ran on one thread whatever was asked for. With 16 threads it is now about 7x faster on a 300-column alignment.
  * **Faster CUDA kernels.** The pseudo-likelihood and edge-gradient kernels were rewritten. Whole runs are 1.4x, 1.8x and 2.2x faster at 75, 300 and 600 columns, with contact matrices that match the old kernels to 1e-5. Reductions use explicit warp synchronisation, as required since Volta.
  * **Builds on current systems.** The build works on RHEL 9-family Linux (glibc 2.34 and later), where it failed at link time over `fsqrt`, and on macOS with Homebrew.
  * **Portable builds.** `-DWITH_NATIVE=OFF` builds a binary that runs on any x86-64 CPU, and the CUDA architectures can be chosen at configure time (see [Compilation](#compilation)).
  * **Continuous integration.** Ubuntu 22.04 and 24.04, Rocky Linux 9 and macOS 14 are built and run on every push. CI has no GPU, so `test/check_gpu.py` compares the GPU and CPU paths locally (see [Testing](#testing)).

## Requirements

To compile from source, you will need:

  * a C compiler with OpenMP support (GCC or Clang; on macOS, Homebrew's `libomp`)
  * [CMake](http://cmake.org/) 3.18 or later
  * Optional: [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 11.0 or later (if you want to compile for the GPU)
  * Optional: `jansson`, `msgpack` and `libuuid` for JSON and MessagePack metadata output, and `ncurses` for the progress display. Each is used when found.

To run CUDA-accelerated computations, you will need an NVIDIA GPU supported by your CUDA Toolkit, the proprietary NVIDIA drivers, and a binary built for that GPU's architecture. See the [NVIDIA CUDA GPU Overview](https://developer.nvidia.com/cuda-gpus) for the compute capability of your card.

### Memory Requirement on the GPU
When doing computations on the GPU, the available memory limits the size of the model you will be able to compute. We recommend at least 2 GB of GPU RAM so you can calculate contacts for big multiple sequence alignments (e.g for N=5000):

	GPU RAM		Lmax	Lmax(pad)
	1 GB		353	291
	2 GB		512	420
	3 GB		635	519
	5 GB		829	676
	6 GB		911	743
	8 GB		1057	861
	12 GB		1302	1059

You can calculate the memory requirements in bytes for L columns and N rows using the following formula:

	4*(4*(L*L*21*21 + L*20) + 23*N*L + N + L*L) + 2*N*L + 1024

For the padded version:

	4*(4*(L*L*32*21 + L*20) + 23*N*L + N + L*L) + 2*N*L + 1024

## Installation
We recommend compiling CCMpred on the machine that should run the computations so that it can be optimized for the appropriate CPU/GPU architecture.

### Downloading
Clone this fork:

	git clone https://github.com/pstansfeld/CCMpred.git

`libconjugrad` is included in the repository, so no submodules need to be fetched.

### Compilation
With the source code ready, run CMake with the default settings; libraries are detected automatically:

	cmake -B build
	cmake --build build -j

You should find the compiled version of CCMpred at `build/bin/ccmpred`. To check if the CUDA libraries were detected, you can run `ldd build/bin/ccmpred` to see if CUDA was linked with the program, or simply run a prediction and check the program's output.

Build options:

  * **CPU tuning.** By default CCMpred is optimized for the build machine (`-march=native`). Use `-DWITH_NATIVE=OFF` to build a portable binary, e.g. for a shared installation on a cluster whose nodes have different CPUs. The portable build is about 6% slower.
  * **CUDA.** CUDA is enabled automatically when a CUDA compiler is found; use `-DWITH_CUDA=OFF` to disable it.
  * **GPU architectures.** By default CUDA code is built for the GPU in the build machine (CMake 3.24 or later), else for compute capabilities 7.0, 7.5, 8.0 and 8.6. To run on other GPUs, list their architectures, e.g. `-DCMAKE_CUDA_ARCHITECTURES="60;70;75;80;86;89;90"`.
  * **Optional libraries.** `-DWITH_CURSES=OFF`, `-DWITH_MSGPACK=OFF` and `-DWITH_JANSSON=OFF` leave out the corresponding feature.

For example, a portable build for a shared installation with several GPU generations:

	cmake -B build -DWITH_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES="60;70;75;80;86;89;90"
	cmake --build build -j

## Usage

	ccmpred [options] input.aln output.mat

The input is one aligned sequence per line, all of the same length, with the query first (no FASTA headers). `scripts/convert_alignment.py` converts other formats. The output is an L x L matrix of contact scores, with average product correction (APC) applied unless `-A` is given.

  * **GPU or CPU.** The GPU is used by default (`-d 0`; pick another device with `-d N`). `-t THREADS` runs on the CPU with that many threads, and `-d -1` forces the CPU.
  * **Weighting and regularisation.** `-w` sets the sequence reweighting identity threshold (default 0.8) and `-l` the pairwise regularisation factor (default 0.2).
  * **Convergence.** `-n` sets the maximum number of iterations (default 50), and `-e` and `-k` set the convergence criterion.
  * **Raw couplings.** `-r RAWFILE` saves the raw couplings, and `-i INIFILE` starts from saved ones.

Run `ccmpred -h` for the full list. CCMpred prints the effective number of sequences (`Beff`). Accuracy depends on alignment depth: when Beff divided by the number of columns is well below 1, the top contacts are mostly noise.

For example:

	build/bin/ccmpred example/1atzA.aln 1atzA.mat                # GPU 0
	build/bin/ccmpred -t 16 example/1atzA.aln 1atzA.mat          # 16 CPU threads

## Testing

`test/test.sh` builds a CPU-only binary in a temporary directory and runs the example alignment, as CI does:

	bash test/test.sh

CI runners have no GPU. After changing CUDA code, check on a machine with one that the GPU and CPU paths agree:

	python3 test/check_gpu.py build/bin/ccmpred [alignment.aln]

It runs 30 iterations on GPU 0 and on the CPU and requires the two contact matrices to correlate at 0.9999 or better, with mostly the same top-L contacts. They are not identical, because the two paths sum in different orders.

## Useful scripts

The `scripts/` subdirectory contains some python scripts you might find useful - please make sure both NumPy and BioPython are installed to use them!

  * `convert_alignment.py` - Use BioPython's `Bio.SeqIO` to convert a variety of alignment formats (FASTA, etc.) into the CCMpred alignment input format
  * `top_couplings.py` - Extract the top couplings from an output contact maps in a list representation

## License
CCMpred is released under the GNU Affero General Public License v3 or later. See LICENSE for more details.

## References

	[1] Ekeberg, M., Lövkvist, C., Lan, Y., Weigt, M., & Aurell, E. (2013).
	    Improved contact prediction in proteins: Using pseudolikelihoods to infer Potts models.
	    Physical Review E, 87(1), 012707. doi:10.1103/PhysRevE.87.012707

	[2] Balakrishnan, S., Kamisetty, H., Carbonell, J. G., Lee, S.-I., & Langmead, C. J. (2011).
	    Learning generative models for protein fold families.
	    Proteins, 79(4), 1061–78. doi:10.1002/prot.22934

	[3] Seemayer, S., Gruber, M., & Söding, J. (2014).
	    CCMpred—fast and precise prediction of protein residue–residue contacts from correlated mutations.
	    Bioinformatics, 30(22), 3128–3130. doi:10.1093/bioinformatics/btu500
