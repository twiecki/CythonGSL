from cython_gsl cimport *
from libc.stdio cimport printf

def main():
	cdef gsl_spmatrix *A = gsl_spmatrix_alloc(5, 4)
	cdef gsl_spmatrix *B, *C
	cdef size_t i, j

	# build the sparse matrix
	gsl_spmatrix_set(A, 0, 2, 3.1)
	gsl_spmatrix_set(A, 0, 3, 4.6)
	gsl_spmatrix_set(A, 1, 0, 1.0)
	gsl_spmatrix_set(A, 1, 2, 7.2)
	gsl_spmatrix_set(A, 3, 0, 2.1)
	gsl_spmatrix_set(A, 3, 1, 2.9)
	gsl_spmatrix_set(A, 3, 3, 8.5)
	gsl_spmatrix_set(A, 4, 0, 4.1)

	print "printing all matrix elements:"
	for i in range(5):
		for j in range(4):
			print "A(%d,%d) = %g" % (i, j, gsl_spmatrix_get(A, i, j))

	printf("matrix in triplet format (i,j,Aij):\n")
	gsl_spmatrix_fprintf(stdout, A, "%.1f")

	# convert to compressed column format
	B = gsl_spmatrix_ccs(A)

	print("\nmatrix in compressed column format:")
	print("i = [ ")
	for i in range(B.nz):
		print "%d, " % B.i[i]
	print "    ]"

	print("p = [ ")
	for i in range(B.size2):
		print "%d, " % B.p[i]
	print "    ]"

	print("d = [ ")
	for i in range(B.nz):
		print "%g, " % B.data[i]
	print "    ]"

	# convert to compressed row format
	C = gsl_spmatrix_crs(A)

	print "\nmatrix in compressed row format:"
	print "i = [ "
	for i in range(C.nz):
		print "%d, " % C.i[i]
	print "    ]"

	print("p = [ ")
	for i in range(C.size1):
		print "%d, " % C.p[i]
	print "    ]"

	print("d = [ ")
	for i in range(C.nz):
		print "%g, " % C.data[i]
	print "    ]"

	gsl_spmatrix_free(A)
	gsl_spmatrix_free(B)
	gsl_spmatrix_free(C)
