// $Id: ReorderCSR.c,v 1.1 2010/11/04 15:35:14 asuzuki Exp asuzuki $ 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

#define EPS2 1.0e-20

#define AMD_ORDERING
#ifdef AMD_ORDERING
#include "amd.h"
#endif // #ifdef AMD_ORDERING

typedef struct {
  int col_ind;
  double val;
} csr_data;

#define AMD      4
#define DESCEND  1

void init_CSR(double *val, int *col_ind, int *row_ptr, int nnz, int n);
void print_CSR(char *st, 
	       double *val, int *col_ind, int *row_ptr, int n);
void make_order_index(int *ordering, int *row_ptr, int n, int descend);
int comp_int(const void *_a, const void *_b);
int comp_col_ind(const void *_a, const void *_b);

void draw_csr(char *buf, int *row_ptr, int *csr_ind, int num_row);
int count_padding(int *nonzeros, int *reordering, int num_row, int block_size);

void makeRgCSR(double *val_new, int *col_ind_new, int *nonzeros, int *grp_ptr,
	       double *val, int *col_ind, int *row_ptr, int group_size, int n);

int countalignedRgCSR(int *row_ptr,  int group_size, int n);

void SpMVCSR(double *y, double *x, double *val, 
	     int *col_ind, int *row_ptr, int num_row);

void  SpMVRgCSR(double *y, double *x, 
		double *val, int *col_ind, int *nonzeros, int *grp_ptr, 
		int block_size, int n);

void reorder_csr_matrix(double *val_new, int *col_ind_new, int *row_ptr_new, 
			double *val, int *col_ind, int *row_ptr, 
			int *ordering, int *reordering, csr_data *work, 
			int num_row);

int main(int argc, char **argv)
{
  double *val, *val_coo, *val_new;
  int *col_coo, *row_coo, *col_ind, *row_ptr, *nonzeros;
  int *col_ind_new, *row_ptr_new;
  int *ordering, *reordering;
  double *val_rgcsr, *val_rgcsr_new;
  double *x, *xx, *y, *y_rgcsr, *y_rgcsr_new;

  int *col_ind_rgcsr, *nonzeros_rgcsr, *grp_ptr_rgcsr;
  int *col_ind_rgcsr_new, *nonzeros_rgcsr_new, *grp_ptr_rgcsr_new;
  int num_row, num_col, num_nz, num_nz0;
  int jtmp;
  FILE *fp;
  char in_file[256], out_file[256], buf[256];
  int block_size = 32;
  int max_nonzeros, min_nonzeros, padding;
  double mean_nonzeros;
  int verbose = 0, graph_output = 0;
  int method_ordering = DESCEND;
  int flag_symmetric = 0;
  int c;
  // clear file name
  in_file[0] = out_file[0] = 0;

  while ((c = getopt(argc, argv, 
		     "ADGvg:i:o:")) != EOF) {
    switch(c) {
    case 'G':
      graph_output = 1;
      break;
    case 'i':
      strcpy(in_file, optarg);
      break;
    case 'o':
      strcpy(out_file, optarg);
      break;
    case 'A':
      method_ordering = AMD;
      break;
    case 'D':
      method_ordering = DESCEND;
      break;
    case 'g':
      block_size = atoi(optarg);
      break;
    case 'v':
      verbose = 1;
      break;
    case 'h':
      fprintf(stderr, 
	      "ReorderCSR -h -A -D -v -i [infile] -o [outfile] -g [group_size]\n");
      break;
    }
  }

  if (in_file[0] == 0 || out_file[0] == 0) {
    fprintf(stderr, "matrix file name is incorrect\n");
  }
  if((fp = fopen(in_file, "r")) == NULL) {
    exit(-1);
  }

  while (1) {
    fgets(buf, 256, fp);
    if (strstr(buf, "%%MatrixMarket") != NULL && 
	strstr(buf, "symmetric") != NULL) {
      flag_symmetric = 1;
      if(verbose) {
	printf("symmetric\n");
      }
    }
    if (buf[0] != '%') {
      break;
    }
  }
  sscanf(buf, "%d %d %d", &num_row, &num_col, &num_nz);

  col_coo = (int *)malloc(sizeof(int) * num_nz);
  row_coo = (int *)malloc(sizeof(int) * num_nz);
  val_coo = (double *)malloc(sizeof(double) * num_nz);

  for (int j = 0; j < num_nz; j++) {
    fgets(buf, 256, fp);
    sscanf(buf, "%d %d %lf", &row_coo[j], &col_coo[j], &val_coo[j]);
    // for C index array style starting at 0
    row_coo[j]--;
    col_coo[j]--;
  }
  fclose(fp);

  // count diagonal parts
  num_nz0 = num_nz;
  if (flag_symmetric) {
    num_nz = num_nz * 2;
    int ktmp = 0;
    for (int i = 0; i < num_nz0; i++) {
      if (row_coo[i] == col_coo[i]) {
	ktmp++;
      }
    }
    num_nz -= ktmp;
  }

  col_ind = (int *)malloc(sizeof(int) * num_nz);
  col_ind_new = (int *)malloc(sizeof(int) * num_nz);
  val = (double *)malloc(sizeof(double) * num_nz);
  val_new = (double *)malloc(sizeof(double) * num_nz);
  row_ptr = (int *)malloc(sizeof(int) * (num_row + 1));
  row_ptr_new = (int *)malloc(sizeof(int) * (num_row + 1));
  nonzeros = (int *)malloc(sizeof(int) * num_row);
  ordering = (int *)malloc(sizeof(int) * num_row);
  reordering = (int *)malloc(sizeof(int) * num_row);

  if(verbose) {
    printf("%d %d %d\n", num_row, num_col, num_nz);
  }

  for (int i = 0; i < num_row; i++) {
    nonzeros[i] = 0;
  }
  for (int j = 0; j < num_nz0; j++) {
    nonzeros[row_coo[j]]++;
    if (flag_symmetric) {
      if (row_coo[j] != col_coo[j]) {
	nonzeros[col_coo[j]]++;
      }
    }
  }

  row_ptr[0] = 0;
  for (int i = 0; i < num_row; i++) {
    row_ptr[i + 1] = row_ptr[i] + nonzeros[i];
  }

  for (int i = 0; i < num_row; i++) {
    reordering[i] = i;
  }

  padding = count_padding(nonzeros, reordering, num_row, block_size);
  if(verbose) {
    printf("original:  %d\n", padding);
  }
  // make CSR format
  for (int i = 0; i < num_row; i++) {
    nonzeros[i] = 0;
  }
  for (int j = 0; j < num_nz0; j++) { 
    int ii = row_coo[j];
    int jj = col_coo[j];
    int ktmp = row_ptr[ii] + nonzeros[ii];
    col_ind[ktmp] = jj;
    val[ktmp] = val_coo[j];
    nonzeros[ii]++;
    if (flag_symmetric) {
      if (ii != jj) {
	ktmp = row_ptr[jj] + nonzeros[jj];
	col_ind[ktmp] = ii;
	val[ktmp] = val_coo[j];
	nonzeros[jj]++;
      }
    }
  }

  max_nonzeros = 0;
  for (int i = 0; i < num_row; i++) {
    if (max_nonzeros < nonzeros[i]) {
      max_nonzeros = nonzeros[i];
    }
  }
  csr_data *work;
  work = (csr_data *)malloc(max_nonzeros * sizeof(csr_data));

  // sort column index in each row
  for (int i = 0; i < num_row; i++) {
    int ktmp = 0;
    for (int k = row_ptr[i]; k < row_ptr[i + 1]; k++) {
      work[ktmp].col_ind = col_ind[k];
      work[ktmp].val     = val[k];
      ktmp++;
    }
    qsort(work, nonzeros[i], sizeof(csr_data), comp_col_ind);
    ktmp = 0;
    for (int k = row_ptr[i]; k < row_ptr[i + 1]; k++) {
      col_ind[k] = work[ktmp].col_ind;
      val[k] = work[ktmp].val;
      ktmp++;
    }
  }
  strcpy(buf, in_file);
  strcat(buf, ".ps");

  if (graph_output) {
    draw_csr(buf, row_ptr, col_ind, num_row);
  }

  strcpy(buf, out_file);

  switch(method_ordering) {
#ifdef AMD_ORDERING
  case AMD: 
    {
      double Control [AMD_CONTROL], Info [AMD_INFO];
      
      amd_defaults(Control) ;
      amd_control(Control) ;
      (void)amd_order(num_row, row_ptr, col_ind, reordering, Control, Info);
      // make inverse mapping : old -> new
      if (verbose) {
	amd_info(Info);
      }
      for (int i = 0; i < num_row; i++) {
	ordering[reordering[i]] = i;
      }
      strcat(buf, ".amd.ps");
    }
    break;
#endif
  case DESCEND:
    make_order_index(ordering, row_ptr, num_row, 1);
    for (int i = 0; i < num_row; i++) {
      reordering[ordering[i]] = i;
    }
    strcat(buf, ".descend.ps");
    break;
  }

  // ordering[i] : old -> new, new index with dreasing order of nonzro

  padding = count_padding(nonzeros, reordering, num_row, block_size);

  if(verbose) {
    switch(method_ordering) {
    case AMD:
      printf("amd:       ");
      break;
    case DESCEND:
      printf("descending:");
      break;
    }
    printf("%d\n", padding);
  }
  reorder_csr_matrix(val_new, col_ind_new, row_ptr_new, 
		     val, col_ind, row_ptr, 
		     ordering, reordering, work, num_row);
  
  
  if((fp = fopen(out_file, "w")) == NULL) {
    exit(-1);
  }
  fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n");
  fprintf(fp, "%d %d %d\n", num_row, num_row, row_ptr_new[num_row]);
  for (int i = 0; i < num_row; i++) {
    for (int j = row_ptr_new[i]; j < row_ptr_new[i + 1]; j++) {
      fprintf(fp, "%d %d %g\n", (i + 1), (col_ind_new[j] + 1), val_new[j]);
    }
  }
  fclose(fp);

  if (graph_output) {
    draw_csr(buf, row_ptr_new, col_ind_new, num_row);
  }

  min_nonzeros = num_row;
  mean_nonzeros = 0.0;
  for (int i = 0; i < num_row; i++) {
    mean_nonzeros += (double)nonzeros[i];
    if (min_nonzeros > nonzeros[i]) {
      min_nonzeros = nonzeros[i];
    }
  }
  mean_nonzeros /= (double)num_row;
  if (verbose) {
    printf("max nonzeros = %d mean= %g min = %d\n", 
	   max_nonzeros, mean_nonzeros, min_nonzeros);
  }
}

void print_CSR(char *st, 
	       double *val, int *col_ind, int *row_ptr, int n)
{
  printf("[ %s ]\n", st);
  for (int i = 0; i < n; i++) {
    printf("%d : [%d] ", i,  row_ptr[i + 1] -  row_ptr[i]);
    for (int k = row_ptr[i]; k < row_ptr[i + 1]; k++) {
      printf(" %g:%d ", val[k], col_ind[k]);
    }
    printf("\n");
  }
}


void make_order_index(int *ordering, int *row_ptr, int n, int descend)
{
  int *slices, *slice_offset;
  int mn;
  
  // find maximum nonzeros from all rows
  mn = 0;
  for (int i = 0; i < n; i++) {
    int non_zeros = row_ptr[i + 1] - row_ptr[i];
    if (mn < non_zeros) {
      mn = non_zeros;
    }
  }
  // prepare working array : this suppose row without element
  slices = (int *)malloc(sizeof(int) * (mn + 1));
  slice_offset = (int *)malloc(sizeof(int) * (mn + 1));
  for (int i = 0; i <= mn; i++) {
    slices[i] = 0;
    slice_offset[i] = 0;
  }
  // slices[i] keeps number of indices of rows whos width is i
  for (int i = 0; i < n; i++) {
    int non_zeros = row_ptr[i + 1] - row_ptr[i];
    slices[non_zeros]++;
  }
  // making blocks in decreasing order of nonzeros
  if (descend) {
    slice_offset[mn] = 0;
    for (int i = mn - 1; i >= 0; i--) {
      slice_offset[i] = slice_offset[i + 1] + slices[i + 1];
    }
  }
  else {
    slice_offset[0] = 0;
    for (int i = 0; i < mn; i++) {
      slice_offset[i + 1] = slice_offset[i] + slices[i];
    }
  }
  
  // this keeps original ordeing wihtin a block
  for (int i = 0; i < n; i++) {
    int non_zeros = row_ptr[i + 1] - row_ptr[i];
    ordering[i] = slice_offset[non_zeros]++;
  }

  free(slices);
  free(slice_offset);
  
}
 
int  comp_int(const void *_a, const void *_b) {
  // cast to deal with arguments defined as void *
  int a = *(int *)_a;
  int b = *(int *)_b;

  if (a < b) {
    return -1;
  } else if (a > b) {
    return 1;
  }
  else {
    return 0;
  }
}


int  comp_col_ind(const void *_a, const void *_b) {
  // cast to deal with arguments defined as void *
  int a = (*(csr_data *)_a).col_ind;
  int b = (*(csr_data *)_b).col_ind;

  if (a < b) {
    return -1;
  } else if (a > b) {
    return 1;
  }
  else {
    return 0;
  }
}


void draw_csr(char *buf, int *row_ptr, int *col_ind, int num_row)
{
  FILE *fp;

  if((fp = fopen(buf, "w")) == NULL) {
    exit(-1);
  }
  fprintf(fp, "%%!PS-Adobe-3.0 EPSF-3.0\n%%%%BoundingBox: 5 5 395 395\n");
  fprintf(fp, "/rr { %g } def\n",  0.45 * 380.0 / (double)(num_row + 2));
  fprintf(fp, "/n { newpath } def\n");
  fprintf(fp, "/rl { rlineto } def\n");
  fprintf(fp, "/m { moveto } def\n");
  fprintf(fp,"n 10 10 m 380 0 rl 0 380 rl -380 0 rl 0 -380 rl closepath 0.85 setgray fill\n");
  for (int i = 0; i < num_row; i++) {
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
      fprintf(fp,"n %g %g rr 0 360 arc 0 setgray fill\n", 
	      10.0  + (double)col_ind[j] / (double)(num_row + 2) * 380.0,
	      390.0 - (double)i / (double)(num_row + 2) * 380.0);
    }
  }
  fprintf(fp, "showpage\n");
  fclose(fp);
}


int count_padding(int *nonzeros, int *reordering, int num_row, int block_size)
{
  // count artificial zeros
  int padding = 0;

  for (int k = 0; k < num_row; k += block_size) {
    int block_max = 0;
    for (int j = 0; j < block_size; j++) {
      if (k + j >= num_row) {
	break;
      }
      int kj = reordering[k + j];
      if (block_max < nonzeros[kj]) {
	block_max = nonzeros[kj];
      }
    }
    for (int j = 0; j < block_size; j++) {
      if (k + j >= num_row) {
	break;
      }
      int kj = reordering[k + j];
      padding += block_max - nonzeros[kj];
    }
  }

  return padding;
}

int countalignedRgCSR(int *row_ptr,  int group_size, int n)
{
  int aligned_max;

  aligned_max = 0;
  // find maximumn number of nonzeros in each group
  for (int i = 0; i < n; i += group_size) {
    int ntmp = 0;
    for (int k = 0; k < group_size; k++) {
      int ik = i + k;
      if (ik >= n) {
	break;
      }
      int mtmp = row_ptr[ik + 1] -  row_ptr[ik];
      if (ntmp < mtmp) {
	ntmp = mtmp;
      }
    }
    aligned_max += ntmp * group_size;
  }
  return aligned_max;
}

void makeRgCSR(double *val_new, int *col_ind_new, int *nonzeros, int *grp_ptr,
	       double *val, int *col_ind, int *row_ptr, int group_size, int n)
{
  int jtmp;

  jtmp = 0;
  grp_ptr[0] = 0;
  for (int i = 0; i < n; i+= group_size) {
   int current_group = group_size;
   if (i + group_size > n) {
     current_group = n % group_size;
   }
   int ntmp = 0;
   for (int k = 0; k < current_group; k++) {
     int ik = i + k;
     int mtmp = row_ptr[ik + 1] -  row_ptr[ik];
      if (ntmp < mtmp) {
	ntmp = mtmp;
      }
   }
   int ig = i / group_size;
   if (ig < (n / group_size + (n % group_size != 0) - 1)) {
    grp_ptr[ig + 1] = grp_ptr[ig] + ntmp * group_size;  
   }
   for (int j = 0; j < ntmp; j++) {
     for (int k = 0; k < current_group; k++) {
       int ik = i + k;
       if (j < (row_ptr[ik + 1] - row_ptr[ik])) {
	 col_ind_new[jtmp] = col_ind[row_ptr[ik] + j];
	 val_new[jtmp] = val[row_ptr[ik] + j];
       }
       else {
	 col_ind_new[jtmp] = (-1);
	 val_new[jtmp] = 0.0;
       }
       jtmp++;
     } // loop : k
   }  // loop : j
  }

  for (int i = 0; i < n; i++) {
    nonzeros[i] = row_ptr[i + 1] -  row_ptr[i];
  }
}

void SpMVCSR(double *y, double *x, double *val, 
	     int *col_ind, int *row_ptr, int num_row)
{
  double stmp;
  for (int i = 0; i < num_row; i++) {
    stmp = 0.0;
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
      stmp += x[col_ind[j]] * val[j];
    }
    y[i] = stmp;
  }
}
 
void  SpMVRgCSR(double *y, double *x, 
		double *val, int *col_ind, int *nonzeros, int *grp_ptr, 
		int block_size, int n)
{
  int num_blocks = n / block_size + (n % block_size != 0);
  for (int j = 0; j < num_blocks; j++) {
    for (int k = 0; k < block_size; k++) {
      int irow = j * block_size + k;
      if (irow >= n) {
	return;
      }
      int ptr = grp_ptr[j] + k;
      int crnt_grp_size = block_size;
      if ((j + 1) * block_size > n) {
	crnt_grp_size = n % block_size;
      }
      double stmp = 0.0;
      for (int i = 0; i < nonzeros[irow]; i++) {
	stmp += val[ptr] * x[col_ind[ptr]];
	ptr += crnt_grp_size;
      }
      y[irow] = stmp;
    }
  }
}

void reorder_csr_matrix(double *val_new, int *col_ind_new, int *row_ptr_new, 
			double *val, int *col_ind, int *row_ptr, 
			int *ordering, int *reordering, csr_data *work, 
			int num_row)
{
  // csr_data *work is allocated as max_j (row_ptr[j + 1] - row_ptr[j]) sized
  int jtmp = 0;
  row_ptr_new[0] = 0;
  for (int i = 0; i < num_row; i++) {
    int j = reordering[i];
    int ktmp = 0;
    for (int k = row_ptr[j]; k < row_ptr[j + 1]; k++) {
      work[ktmp].col_ind = ordering[col_ind[k]];
      work[ktmp].val     = val[k];
      ktmp++;
    }
    int itmp = row_ptr[j + 1] - row_ptr[j];
    qsort(work, itmp, sizeof(csr_data), comp_col_ind);
    ktmp = 0;
    for (int k = 0; k < itmp; k++) {
      val_new[jtmp]     = work[k].val;
      col_ind_new[jtmp] = work[k].col_ind;
      jtmp++; 
    }
    row_ptr_new[i + 1] = jtmp;
  }
}
