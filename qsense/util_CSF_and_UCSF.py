import numpy as np
import scipy, time, sys, copy, os
from openfermion import FermionOperator, hermitian_conjugated, normal_ordered, get_ground_state, number_operator
from openfermion.utils import commutator
from ferm_utils import (
    get_on_idx, get_on_vec,op_action_tz_remove_0coef,
    braket_tz, get_S_squared, get_S_z, judge_eigen_on_list, number_ferm_tz,
    op_action_tz_CSF
)
import itertools
from itertools import combinations, accumulate
from scipy.sparse import csr_matrix, lil_matrix
from scipy.optimize import minimize
from optimparallel import minimize_parallel
#from pylanczos import PyLanczos
from joblib import Parallel, delayed

def get_kappa_isig_asigprim(i_sig,a_sig,debug=False):
    """
    i_sig is the spin orbital with spatial orbital i and spinor sig
    a_sig is the spin orbital with spatial orbital a and spinor sig
    The two spinors may not be idenical
    """
    if debug: print(f'in get_kappa_isig_asig')
    if not isinstance(i_sig,str):
        print(f'{i_sig} is not a string. Bombing out')
        sys.exit()
    if not isinstance(a_sig,str):
        print(f'{a_sig} is not a string. Bombing out')
        sys.exit()

    if debug: print(i_sig,a_sig)
    i_spatial = int(i_sig[:-1])
    i_spinor = i_sig[-1]
    a_spatial = int(a_sig[:-1])
    a_spinor = a_sig[-1]
    if debug: print(i_spatial,i_spinor,a_spatial,a_spinor)
    if i_spinor == 'a':
        i_spinorb = i_spatial*2
    elif i_spinor == 'b':
        i_spinorb = i_spatial*2+1
    else:
        print(f'Unrecognized spinor for {i_sig}')

    if a_spinor == 'a':
        a_spinorb = a_spatial*2
    elif a_spinor == 'b':
        a_spinorb = a_spatial*2+1
    else:
        print(f'Unrecognized spinor for {a_sig}')

    if debug: print(i_spinorb,a_spinorb)

    kappa_isig_asigprim = get_kappa_ia(i_spinorb,a_spinorb)
    return kappa_isig_asigprim

def get_kappa_ia(i,a):
    """
    Read in two indices of spin orbital and return
    kappa_ia = a_a^+ a_i - a_i^+ a_a
    """

    term = ((a,1),(i,0))
    coef = 1.0
    kappa_ia = FermionOperator(term,coef)

    kappa_ia -= hermitian_conjugated(kappa_ia)

    return kappa_ia

def get_kappa_ijab(i,j,a,b):
    """
    Read in four indices of spin orbitals i, j, a, b, and return
    kappa_ijab = a_a^+ a_b^+ a_j a_i - h.c.
    """

    term = ((a,1),(b,1),(j,0),(i,0))
    coef = 1.0
    kappa_ijab  = FermionOperator(term,coef)
    kappa_ijab -= hermitian_conjugated(kappa_ijab)

    return kappa_ijab

def get_kappa_ijab_spaspn(isig,jsig,asig,bsig):
    """
    Read in four indices of spatial-spin orbitals ia, jb, aa, bb, etc. and return
    the corresponding kappa_ijab = a_aa^+ a_bb^+ a_jb a_ia - h.c. etc.
    """


    i_spinorb = ia_to_2i_ib_to_2iplus1(isig)
    j_spinorb = ia_to_2i_ib_to_2iplus1(jsig)
    a_spinorb = ia_to_2i_ib_to_2iplus1(asig)
    b_spinorb = ia_to_2i_ib_to_2iplus1(bsig)


    kappa_ijab = get_kappa_ijab(i_spinorb,j_spinorb,a_spinorb,b_spinorb)

    return kappa_ijab

def ia_to_2i_ib_to_2iplus1(i_sig):
    """
    Convert ia to 2i and ib to 2i+1, i.e., convert spatial-spin orbital index to
    spin orbital index
    """

    i_spatial = int(i_sig[:-1])
    i_spinor = i_sig[-1]
    if i_spinor == 'a':
        i_spinorb = 2*i_spatial
    elif i_spinor == 'b':
        i_spinorb = 2*i_spatial+1
    else:
        print(f'Unrecognized spinor for {i_sig}')

    return i_spinorb

def get_Tia_00(i_spatial,a_spatial):
    """
    Read in two spatial orbitals indexing integers i and a and return
    T_ia_00 = kappa_ialpha_aalpha + kappa_ibeta_abeta
    """

    ia = str(i_spatial)+'a'
    aa = str(a_spatial)+'a'
    ib = str(i_spatial)+'b'
    ab = str(a_spatial)+'b'

    Tia_00  = get_kappa_isig_asigprim(ia,aa) + get_kappa_isig_asigprim(ib,ab)
    Tia_00 *= np.sqrt(0.5)
    return Tia_00

def get_Tia_1m(i_spatial,a_spatial):
    """
    Read in two spatial orbitals indexing integers i and a and return
    T_ia_1p1 = kappa_ibeta_aalpha
    T_ia_1_0 = 1/sqrt(2) ( kappa_ibeta_abeta - kappa_ialpha_aalpha )
    T_ia_1m1 = -kappa_ialpha_abeta
    """

    ia = str(i_spatial)+'a'
    aa = str(a_spatial)+'a'
    ib = str(i_spatial)+'b'
    ab = str(a_spatial)+'b'

    Tia_1p1 = get_kappa_isig_asigprim(ib,aa)
    Tia_1_0 = get_kappa_isig_asigprim(ib,ab) - get_kappa_isig_asigprim(ia,aa)
    Tia_1_0 *= np.sqrt(0.5)
    Tia_1m1 = -get_kappa_isig_asigprim(ia,ab)

    return Tia_1p1, Tia_1_0, Tia_1m1

def get_Tiiaa_00(i_spatial,a_spatial):
    """
    Read in two spatial orbitals indexing integers i and a and return
    Tiiaa_00 = kappa_ialpha_ibeta_aalpha_abeta
    """

    ia = str(i_spatial)+'a'
    aa = str(a_spatial)+'a'
    ib = str(i_spatial)+'b'
    ab = str(a_spatial)+'b'

    Tiiaa_00 = get_kappa_ijab_spaspn(ia,ib,aa,ab)

    return Tiiaa_00

def get_sqrt_half_Tiiaa_00_plus_Tiibb_00(i_spatial,a_spatial,b_spatial):
    """
    Read in spatial orbitals i, a and b and return
    1/sqrt(2) * (Tiiaa + Tiibb)
    """

    ia = str(i_spatial)+'a'
    aa = str(a_spatial)+'a'
    ib = str(i_spatial)+'b'
    ab = str(a_spatial)+'b'
    ba = str(b_spatial)+'a'
    bb = str(b_spatial)+'b'

    Tiiaa_00 = get_kappa_ijab_spaspn(ia,ib,aa,ab)
    Tiibb_00 = get_kappa_ijab_spaspn(ia,ib,ba,bb)

    res = np.sqrt(0.5)*(Tiiaa_00 + Tiibb_00)
    res = normal_ordered(res)
    res.compress()
    return res


def get_E_ia(i_spinorb,a_spinorb):
    """
    Read in spin orbital indices i and a, and return the excitaiton
    operator E_i^a = a_a^+ a_i
    """

    term = ((a_spinorb,1),(i_spinorb,0))
    coef = 1.0

    E_ia = FermionOperator(term,coef)

    return E_ia

def get_pair_excite(i_spatial,a_spatial):
    """
    Read in two spatial orbitals indexing integers i and a and return
    E_ia^aa * E_ib^ab
    """

    ia = 2*i_spatial
    aa = 2*a_spatial
    ib = ia + 1
    ab = aa + 1
    E_ia_aa = get_E_ia(ia,aa)
    E_ib_ab = get_E_ia(ib,ab)

   #print(E_ia_aa,E_ib_ab,E_ia_aa*E_ib_ab)

    return E_ia_aa*E_ib_ab

def print_matrix(matrix,n_per_group=6):

    nrow = matrix.shape[0]
    ncolumn = matrix.shape[1]

    ngroup = ncolumn // n_per_group
    nleft = ncolumn - ngroup*n_per_group

    for igroup in range(ngroup):
        imin = igroup*n_per_group
        imax = imin + n_per_group
        for row in matrix[:,imin:imax]:
            print(" ".join(f"{num:12.6f}" for num in row))

        print("")

    if nleft > 0:
        if ngroup == 0:
            imin = 0
        else:
            imin = imax
        imax = ncolumn
        for row in matrix[:,imin:imax]:
            print(" ".join(f"{num:12.6f}" for num in row))

        print("")

def print_eigen_solution(eigen_values,eigen_vectors):

    nsolut = len(eigen_values)
    n_per_group = 5
    ngroup = nsolut // n_per_group
    nleft = nsolut - ngroup*n_per_group
   #print(nsolut,ngroup,nleft)
    for igroup in range(ngroup):
        imin = igroup*n_per_group
        imax = imin + n_per_group
        print(" ".join(f"{num:12.6f}" for num in eigen_values[imin:imax]))
        print("")

        for row in eigen_vectors[:,imin:imax]:
            print(" ".join(f"{num:12.6f}" for num in row))

        print("")

    if nleft > 0:
        if ngroup == 0:
            imin = 0
        else:
            imin = imax
        imax = nsolut
        print(" ".join(f"{num:12.6f}" for num in eigen_values[imin:imax]))
        print("")

        for row in eigen_vectors[:,imin:imax]:
            print(" ".join(f"{num:12.6f}" for num in row))

        print("")

def make_and_apply_U(list_ampld,list_genmat,list_mp2_Ecorr,Hmat,vec_before_U,debug=False):
    """
    Read in rotational amplitudes, generator matrices, Hamiltonian matrix, and an initital vector
    in a space, construct the U matrix, act the rotational U matrix on the
    vector, and calculate the average energy of the resultant state
    """

    assert len(list_ampld) == len(list_genmat)

    if debug: print(f'\nStep Ecorr.        Step MP2 Ecorr.        Accum. Step Ecorr.       Accum. Ecorr.')
    n_basis = Hmat.shape[0]
    Umat = np.eye(n_basis)
    E_input_state = vec_before_U.transpose()@Hmat@vec_before_U
    sum_Ecorr_separate_generator = 0.0
    for igen,genmat in enumerate(list_genmat):
        norm_genmat = np.linalg.norm(genmat)
        if np.isclose(norm_genmat,0.0):
            if debug:
                print(f'Skipping the 0 generator for igen {igen}')
            continue
       #print(f'igen = {igen}, {list_ampld[igen]}')
       #print(f'igen = {igen}, genmat:')
       #print(genmat)
       #tstart = time.perf_counter()
       #exp_genmat_quick = quick_expon_genmat(list_ampld[igen],genmat,False)
       #tend   = time.perf_counter()
       #print(f'Time to get exp_genmat_quick: {tend - tstart}')
       #print(f'exp_genmat_quick:')
       #print_matrix(exp_genmat_quick)
       #tstart = time.perf_counter()
        exp_genmat = scipy.linalg.expm(list_ampld[igen]*genmat)
       #tend   = time.perf_counter()
       #print(f'Time to get exp_genmat: {tend - tstart}')
       #assert np.isclose(np.linalg.norm(exp_genmat_quick - exp_genmat),0.0)
       #print(f'exp_genmat:')
       #print_matrix(exp_genmat)
       #print(f'Norm of deviation matrix: {np.linalg.norm(exp_genmat_quick - exp_genmat)}')
        Umat @= exp_genmat
        if debug:
            E_step = vec_before_U.transpose()@exp_genmat.transpose()@Hmat@exp_genmat@vec_before_U
            sum_Ecorr_separate_generator += E_step - E_input_state
            E_update = vec_before_U.transpose()@Umat.transpose()@Hmat@Umat@vec_before_U
            E_mp2_corr_1_gen = np.sum(np.array(list_mp2_Ecorr[igen]))
           #print(f'E_corr induced by the {igen}-th generator itself: {E_step - E_input_state}')
           #print(f'Sum of independent correlation energies by individual generator: {sum_Ecorr_separate_generator}')
            if debug:
                print(E_step - E_input_state,E_mp2_corr_1_gen,sum_Ecorr_separate_generator,E_update-E_input_state)

       #print(f'Umat, igen = {igen}')
       #print(Umat)

   #print(f'Umat in E_func_of_rotamp:')
   #print(Umat)
    vec_after_U = Umat@vec_before_U
    E_avrg = vec_after_U.transpose()@Hmat@vec_after_U

    return E_avrg, Umat, vec_after_U

def quick_expon_genmat(theta,genmat,debug=False):
    """
    Read in a generator matrix and returns the exponential of the generator matrix
    This code is fast when genmat is a sparse antihermitian matrix.
    It HOWEVER turns out not as fast as the built-in expm code. OK, this code is
    not used momentarily.
    """

    if debug: print('In quick_expon_genmat')

    ndim = genmat.shape[0]

   #expmat = np.zeros([ndim,ndim])
    expmat = np.eye(ndim)

   #assert genmat is a real antihermitian matrix
    assert np.isclose(np.sum((genmat.transpose() + genmat)**2),0.0)
    rows, cols = np.nonzero(genmat)
    if debug: print(rows,cols,len(rows),len(cols))

    row_col_considered = []
    for irow, row in enumerate(rows):
        col = cols[irow]
        if row in row_col_considered or col in row_col_considered:
            if debug: print(f'Row {row} or Col {col} of nonzero item {irow} has been considered')
            continue
        row_considering = []
        item_match = np.where(rows == row)[0]
       #print(f'item_match for row = {row},{item_match}')
        row_considering.append(row)
        for item in item_match:
           #print(f'item = {item}')
            rol_prm = cols[item]
            if rol_prm not in row_considering: row_considering.append(rol_prm)

        if debug: print(f'row_considering after 1st sweeping: {row_considering}')

        nrow_old = len(row_considering)
        nrow_new = -1
        nsweep = 1
        lsweep = True
        while lsweep:
            nsweep += 1
            for row_prm in row_considering:
                item_match = np.where(rows == row_prm)[0]
                for item in item_match:
                    rol_dprm = cols[item]
                    if rol_dprm not in row_considering: row_considering.append(rol_dprm)
            nrow_new = len(row_considering)
            if nrow_new == nrow_old:
                lsweep = False
            else:
                nrow_old = nrow_new

        if debug: print(f'row_considering after {nsweep} sweepings: {row_considering}')
        row_col_considered.extend(row_considering)

       #nsubdim = len(row_considering)
       #subgenmat = genmat[row_considering[:, None],row_considering]
        subgenmat = genmat[np.ix_(row_considering,row_considering)]
       #print('sub genmat:')
       #print_matrix(subgenmat)

        subexpmat = scipy.linalg.expm(theta*subgenmat)
        if debug:
            print('sub expmat:')
            print_matrix(subexpmat)

        expmat[np.ix_(row_considering,row_considering)]=subexpmat

        if debug:
            print('Updated full expmat')
            print_matrix(expmat)

    return expmat

def make_iapair_genmat_fast(list_iapair,list_basis,list_iapair_st_pairs,debug=False):
    """
    Make list of generator matrices for mp2 ia pairs
    """
    if debug: print('\nIn make_iapair_genmat_fast')

    n_basis = len(list_basis)

    if debug:
        print('list_iapair in make_iapair_genmat_fast')
        for item in list_iapair:
            print(item)

    list_genmat = []
    for ipairs, pairs in enumerate(list_iapair):
       #if len(pairs) > 2:
       #    print(pairs)
       #    print(f'Now the fast code only supports axial symmetr with degeneracy <= 2. Bombing out!')
       #    sys.exit()
        genmat = csr_matrix((n_basis,n_basis))
        for pair in pairs:
            lfound = False
            lswap = False
            for item in list_iapair_st_pairs:
                if item[0] == pair:
                    lfound = True
                    states_coupled = item[1:]
                    break
                elif item[0][0] == pair[1] and item[0][1] == pair [0]:
                    lfound = True
                    states_coupled = item[1:]
                    lswap = True
           #if not lfound:
           #    print(f'{pair} not found in list_iapair_st_pairs:')
           #    for item in list_iapair_st_pairs:
           #        print(item)
           #    print('Bombing out!')
           #    sys.exit()
           #lfound = False is very normal. The mp2 ia pair may involve SOMO in the states.
           #So, we simply continue to the next mp2 pair
            if not lfound:
                list_genmat.append(genmat)
                continue
            if debug:
                print(f'The following state pairs: {states_coupled}')
                print(f'are coupled by pair excitations {pair}')
            for states_pair in states_coupled:
                state_high = states_pair[0]
                state_low  = states_pair[1]
                if state_high <= state_low:
                    print(f'state_high <= state_low, {states_pair}. Bombing out!')
                    sys.exit()
                if not lswap:
                    genmat[state_high,state_low ] =  1.0
                    genmat[state_low ,state_high] = -1.0
                else:
                    genmat[state_high,state_low ] = -1.0
                    genmat[state_low ,state_high] =  1.0

            if debug:
                print(f'genmat for exxcitation pair {pair}')
                print(genmat)



            list_genmat.append(genmat)

    if debug:
       #print(f'list_iapair     : {list_iapair}')
        list_genmat_check = make_iapair_genmat(list_iapair,list_basis)
        if not len(list_genmat) == len(list_genmat_check):
            print(f'Inconsistent dimensions of list_genmat and list_genmat_check')
            print(len(list_genmat),len(list_genmat_check))
            sys.exit()

        for ii,genmat_check in enumerate(list_genmat_check):
            genmat_sparse = list_genmat[ii]
            genmat_check_sparse = csr_matrix(genmat_check)
            norm_diff = scipy.sparse.linalg.norm(genmat_sparse - genmat_check_sparse)
            if not np.isclose(norm_diff,0.0):
                print(f'Not identical genmats for {ii}, {list_iapair[ii]}')
                print('genmat_sparse:')
                print(genmat_sparse)
                print('genmat_check_sparse:')
                print(genmat_check_sparse)
                print(f'list_iapair_st_pairs:')
                print(list_iapair_st_pairs)
                print('Bombing out!')
                sys.exit()

    return list_genmat

def make_Umat_decomp_genmat(list_genmat,list_ampld,debug=False):
    """
    Construct U matrix using decomposed genmats
    """

    if debug: print('\nIn make_Umat_decomp_genmat')

    if len(list_genmat) != len(list_ampld):
        print(f'Inconsistent dimensions in list_genmat and list_ampld: {len(list_genmat),len(list_ampld)}')
        print(list_genmat, list_ampld)
        sys.exit()
    if len(list_genmat) == 0: #return a [1] matrix
        Umat = csr_matrix(np.eye(1))
        return Umat

   #tic = time.perf_counter()
    for ii, decomp_genmat in enumerate(list_genmat):
        [list_unique_eigval,list_prj_genmat,eigvec,list_start_end_ind] = decomp_genmat
        theta = list_ampld[ii]
        Umat_1theta = make_analytical_U_decomp_genmat(theta,list_unique_eigval,list_prj_genmat,eigvec,list_start_end_ind)
        if ii == 0:
            Umat = Umat_1theta
        else:
            Umat = Umat_1theta@Umat
   #toc = time.perf_counter()
   #print(f'Time for looping over decomp_genmat: {toc - tic}')

   #if debug:
   #   #print(f'Umat obtained in make_Umat_decomp_genmat')
   #   #print_matrix(Umat.toarray())
   #    det_Umat = sparse_det(Umat)
   #    if not np.isclose(det_Umat,1.0):
   #        print('Determinant of U is not 1: {det_Umat}')
   #        sys.exit()


    return Umat

def make_Umat_decomp_genmat_joblib(list_genmat,list_ampld,nparal,debug=False):
    """
    A joblib parallel version of make_Umat_decomp_genmat
    """

    n_theta = len(list_ampld)

    if n_theta == 0: #return a [1] matrix
        Umat = csr_matrix(np.eye(1))
        return Umat

    def make_Umat_for_one_decomp_genmat(igenmat,list_genmat,list_ampld):
        [list_unique_eigval,list_prj_genmat,eigvec,list_start_end_ind] = list_genmat[igenmat]
        theta = list_ampld[igenmat]
        Umat_1theta = make_analytical_U_decomp_genmat(theta,list_unique_eigval,list_prj_genmat,eigvec,list_start_end_ind)
        return Umat_1theta

   #Good parallel. Comment out the lines between Good Paralle and End of good parallel,
   #and comment on the line of list_Umat_1theta = Parallel ... return to the code before good parallel.
    def make_Umat_for_chunk_decomp_genmat(list_igenmat,list_genmat,list_ampld):
        list_Umat_1theta = []
        for igenmat in list_igenmat:
            Umat_1theta = make_Umat_for_one_decomp_genmat(igenmat,list_genmat,list_ampld)
            list_Umat_1theta.append(Umat_1theta)

        return list_Umat_1theta

    chunk_size = n_theta // nparal
   #print(f'# of all genmats: {n_theta}, divided by nparal: {nparal}')
   #print(f'# of genmats handled by each core = {chunk_size}')

    list_all_idx = []
    for ii in range(n_theta): list_all_idx.append(ii)
    chunks = []
    for ii in range(0,nparal):
        chunks.append(list_all_idx[ii*chunk_size:(ii+1)*chunk_size])

    chunks[-1] += list_all_idx[nparal*chunk_size:]

    list_list_Umat_1theta = Parallel(n_jobs=nparal)(delayed(make_Umat_for_chunk_decomp_genmat)(chunk,list_genmat,list_ampld) for chunk in chunks)
    list_Umat_1theta = []
    for item in list_list_Umat_1theta: list_Umat_1theta += item

   #End of good parallel

   #list_Umat_1theta = Parallel(n_jobs=nparal)(delayed(make_Umat_for_one_decomp_genmat)(ii,list_genmat,list_ampld) for ii in range(n_theta))
    Umat = list_Umat_1theta[0]
    for Umat_1theta in list_Umat_1theta[1:]:
        Umat = Umat_1theta@Umat

    return Umat


def sparse_det(A):
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")

    if A.shape[0] <= 500:
        return scipy.linalg.det(A.toarray())

    lu = scipy.sparse.linalg.splu(A)
    det_A = np.prod(lu.diags()) * (-1)**lu.perm_r.size

    return det_A

def make_analytical_U_decomp_genmat(theta,list_unique_eigval,list_prj_genmat,eigvec,list_start_end_ind):
    """
    Make exp(theta*genmat) using decomposed genmat
    """

    ndim = eigvec.shape[0]
    exp_genmat_anl = csr_matrix((ndim,ndim))
    for ival, unival in enumerate(list_unique_eigval):
        [istart, iend] = list_start_end_ind[ival]
        sub_eigvec = eigvec[:,istart:iend+1]
        sub_eigvec = csr_matrix(sub_eigvec)
        prj_genmat = list_prj_genmat[ival]
        if np.isclose(unival,0.0):
            assert np.isclose(scipy.sparse.linalg.norm(prj_genmat),0.0)
            exp_genmat_anl += sub_eigvec @ sub_eigvec.transpose()
        else:
            ndim_sub = prj_genmat.shape[0]
            sub_expgenmat = scipy.sparse.identity(ndim_sub,format="csr")
            sub_expgenmat += prj_genmat*np.sin(np.sqrt(-unival)*theta)
            sub_expgenmat -= prj_genmat@prj_genmat*(np.cos(np.sqrt(-unival)*theta)-1.0)
            sub_expgenmat_numrc = scipy.linalg.expm(np.sqrt(-unival)*theta*prj_genmat.toarray())
            exp_genmat_anl += sub_eigvec@sub_expgenmat@sub_eigvec.transpose()

    return exp_genmat_anl

def make_analytical_dU_decomp_genmat(theta,list_unique_eigval,list_prj_genmat,eigvec,list_start_end_ind):
    """d/dtheta of exp(theta*genmat), using the SAME spectral decomposition as
    make_analytical_U_decomp_genmat.

    exp(theta G) on each nonzero-eigenvalue block is
        I + P sin(w t) - P^2 (cos(w t) - 1),   w = sqrt(-unival),  P = prj_genmat
    so its theta-derivative is
        w [ P cos(w t) + P^2 sin(w t) ].
    The unival~0 block of exp is a constant projector, so its derivative is 0.
    """
    ndim = eigvec.shape[0]
    dexp_genmat_anl = csr_matrix((ndim,ndim))
    for ival, unival in enumerate(list_unique_eigval):
        if np.isclose(unival,0.0):
            continue   # constant projector block -> derivative is zero
        [istart, iend] = list_start_end_ind[ival]
        sub_eigvec = csr_matrix(eigvec[:,istart:iend+1])
        prj_genmat = list_prj_genmat[ival]
        w = np.sqrt(-unival)
        sub_dexp = prj_genmat*(w*np.cos(w*theta)) \
                 + (prj_genmat@prj_genmat)*(w*np.sin(w*theta))
        dexp_genmat_anl += sub_eigvec@sub_dexp@sub_eigvec.transpose()

    return dexp_genmat_anl

def make_and_apply_U_matrix(list_genmat,list_ampld,l_feedin_inivec=False,vec_feed_in=None,debug=False):
    """
    Given a list of generator matrices and the corresponding amplitudes,
    make the exponential matrix, U = prod_I exp (theta_I gen_I)
    """
    if debug: print('\nIn make_U_matrix')

    if debug:
        print(f'U amplitudes:')
        print(list_ampld)

    ndim = list_genmat[0].shape[0]
    expmat = np.eye(ndim)
    for ii, ampld in enumerate(list_ampld):
        _gm = list_genmat[ii]
        genmat = _gm.toarray() if scipy.sparse.issparse(_gm) else np.asarray(_gm)
        norm_genmat = scipy.sparse.linalg.norm(_gm) if scipy.sparse.issparse(_gm) else np.linalg.norm(genmat)
        if np.isclose(norm_genmat,0.0): continue
        if debug:
           #det_genmat = scipy.linalg.det(genmat)
           #if not np.isclose(det_genmat,0.0):
           #    print(f'genmat does not have a determiannt 1, {det_genmat}')
           #    print_matrix(genmat)
           #    print('Bombing out')
           #    sys.exit()
            norm_to_be_0 = np.linalg.norm(genmat + genmat.transpose())
            if not np.isclose(norm_to_be_0,0.0):
                print(f'genmat {ii} is not antisymmetric:')
                print_matrix(genmat)
                print('genmat + genmat.transpose')
                print_matrix(genmat + genmat.transpose())
                print('Bombing out!')
                sys.exit()
       #tic = time.perf_counter()
        expmat_one_theta = scipy.linalg.expm(ampld*genmat)
       #toc = time.perf_counter()
       #print(f'Time for numerical expmat calculation: {toc - tic}')
       #See whether we can use analytical formula for exp(G)
       #genmat_sparse = list_genmat[ii]
       #tic = time.perf_counter()
       #l_mat_power_equals_scaled_mat, scal_fac = mat_powers_equals_scaled_mat_2nd(genmat_sparse,3)
       #if l_mat_power_equals_scaled_mat:
       #    if scal_fac > 0.0:
       #        print('G^3 = s*G but with s positive: {scal_fac}. Bombing out!')
       #        sys.exit()
       #    else:
       #        genmat_normalized = np.sqrt(-1.0/scal_fac)*genmat_sparse
       #        if not mat_powers_equals_scaled_mat(genmat_normalized,3,-1.0):
       #            print(f'genmat_normalized does not pass mat_powers_equals_scaled_mat')
       #            sys.exit()
       #        scaled_theta = ampld*np.sqrt(-scal_fac)
       #        expmat_one_theta_analytic = np.sin(scaled_theta)*genmat_normalized -\
       #                                    (np.cos(scaled_theta) - 1.0)*genmat_normalized@genmat_normalized
       #        expmat_one_theta_analytic = scipy.sparse.identity(ndim) + expmat_one_theta_analytic
       #       #if not np.isclose(scipy.sparse.linalg.norm(expmat_one_theta_analytic - csr_matrix(expmat_one_theta)),0.0):
       #       #    print('Analytical expm test failed')
       #       #    print(scipy.sparse.linalg.norm(expmat_one_theta_analytic - csr_matrix(expmat_one_theta)))
       #       #    print(expmat_one_theta_analytic - csr_matrix(expmat_one_theta))
       #       #    sys.exit()
       #toc = time.perf_counter()
       #print(f'Time for analytical expmat calculation: {toc - tic}')
        det_expmat_one_theta = scipy.linalg.det(expmat_one_theta)
       #print(f'det_expmat_one_theta: {det_expmat_one_theta}, iampld: {ii}')
        if not np.isclose(det_expmat_one_theta,1.0):
            print(f'Exponentiation of one theta*genmat is not 1')
            print(f'theta = {ampld}')
            print(f'genmat dimension: {ndim}')
            for i in range(ndim):
                for j in range(i,ndim):
                    if not np.isclose(genmat[i,j],-genmat[j,i]):
                        print('The genmat is not an antisymmetric matrix')
                        print(f'{i,j,genmat[i,j],genmat[j,i]}')
            sys.exit()
        expmat = expmat_one_theta@expmat
        det_expmat = scipy.linalg.det(expmat)
       #print(f'Accummulated det_expmat: {det_expmat}, iampld: {ii}')
       #expmat = scipy.linalg.expm(ampld*genmat)@expmat
       #if debug:
       #    print(f'genmat {ii}')
       #    print_matrix(genmat)
       #    print('\nUpdated U')
       #    print_matrix(expmat)

   #det_expmat = scipy.linalg.det(expmat)
   #if not np.isclose(det_expmat,1.0):
   #    print(f'expmat does not have a determinant 1, {det_expmat}')
   #    print(f'Shape of expmt: {expmat.shape}')
   #   #print_matrix(expmat)
   #    print('Bombing out')
   #    sys.exit()
   #   #print('Continue without bombing out. The resultant vector will be normalized')
   #The determinant test above is turned off because determinant may not be accurately
   #claculated for large dimension square matrix
    zero_matrix = expmat@expmat.transpose() - np.eye(ndim)
    resid_norm = np.linalg.norm(zero_matrix)
    if not np.isclose(resid_norm,0.0):
        print(f'expmat is not orthogonal. Residule norm: {resid_norm}. Bombing out!')
        sys.exit()

    if not l_feedin_inivec:
        vec_x_by_U = np.zeros([ndim])
        vec_x_by_U[0] = 1.0
    else:
        vec_x_by_U = vec_feed_in

    U_x_vec = expmat@vec_x_by_U
    if debug:
        print(f'U mat in make_and_apply_u_matrix')
        print_matrix(expmat)

   #Normalize U_x_vec in case expmat's determinant is far away from 1
    U_x_vec = U_x_vec / np.linalg.norm(U_x_vec)

    if debug:
        print(f'U_x_vec: {U_x_vec}')

    return expmat,U_x_vec

def mat_powers_equals_scaled_mat(mat_sparse,npow,scal_fac):
    """
    See whether the n-th power a sparse matrix is equal to scal_fac*matrix
    """

    ndim = mat_sparse.shape[0]
    mat_npow = scipy.sparse.identity(ndim)
    for ii in range(npow):
        mat_npow = mat_npow@mat_sparse

    return np.isclose(scipy.sparse.linalg.norm(mat_npow - scal_fac*mat_sparse),0.0)

def mat_powers_equals_scaled_mat_2nd(mat_sparse,npow):
    """
    See whether n-th power of a sparse square matrix is equal to scal_fac*matrix and
    returns the scaling factor
    """

   #print(f'nonzero at: {mat_sparse.nonzero()}')
    nonzero_places = mat_sparse.nonzero()
    if len(nonzero_places[0]) == 0:
        print(f'All zero matrix should have been screened off before entering mat_powers_equals_scaled_mat_2nd')
        print('Bombing out!')
        sys.exit()
    nonzero_rows = nonzero_places[0]
    nonzero_cols = nonzero_places[1]
    row_set = list(set(nonzero_rows))
    col_set = list(set(nonzero_cols))
   #print(row_set,col_set)
    assert len(row_set) == len(col_set)
    ndim = len(row_set)
   #print(ndim)
    mat_reduced = csr_matrix((ndim,ndim))
   #print(mat_reduced.shape)
    for ii in range(len(nonzero_rows)):
        irow = row_set.index(nonzero_rows[ii])
        icol = col_set.index(nonzero_cols[ii])
        mat_reduced[irow,icol] = mat_sparse[nonzero_rows[ii],nonzero_cols[ii]]
   #print(mat_reduced)
   #print(mat_sparse)
    mat_npow = scipy.sparse.identity(ndim)
    for ii in range(npow):
        mat_npow = mat_npow@mat_reduced

    if np.isclose(scipy.linalg.det(mat_reduced.toarray()),0.0):
       #print(f'Singular mat_sparse detected. Bombing out!')
       #print(mat_reduced)
       #sys.exit()
        return False, None

    mat_inv = scipy.sparse.linalg.inv(mat_reduced)
    test_mat = mat_inv@mat_npow
    scal_fac = test_mat[0,0]
    if np.isclose(scipy.sparse.linalg.norm(test_mat - scal_fac*scipy.sparse.identity(ndim)),0.0):
        return True, scal_fac
    else:
        return False, None

def make_iapair_genmat(list_iapair,list_basis):
    """
    Create the generator matrices for Tiiaa_00 operators
    for a multi-electronic basis set
    """

    list_genmat = []
    n_basis = len(list_basis)
   #print(f'n_basis = {n_basis}')
    for iampld in range(len(list_iapair)):
       #print(iampld,list_iapair[iampld])
        ex_op = FermionOperator()
        for ia_pair in list_iapair[iampld]:
            i = ia_pair[0]
            a = ia_pair[1]
            Tiiaa_00 = get_Tiiaa_00(i,a)
            ex_op += Tiiaa_00

        generator_matrix = np.zeros([n_basis,n_basis])
        for ibasis,basis_bra in enumerate(list_basis):
            SDs_bra   = basis_bra[0]
            coefs_bra = basis_bra[2]
            for jbasis in range(ibasis+1,n_basis):
                basis_ket = list_basis[jbasis]
                SDs_ket   = basis_ket[0]
                coefs_ket = basis_ket[2]

                dsum = 0.0
                for iSD_bra in range(len(SDs_bra)):
                    onl = SDs_bra[iSD_bra]
                    coefl = coefs_bra[iSD_bra]
                    for jSD_ket in range(len(SDs_ket)):
                        onr = SDs_ket[jSD_ket]
                        coefr = coefs_ket[jSD_ket]
                        dsum += coefl*coefr*braket_tz(onl,onr,ex_op)

                generator_matrix[ibasis,jbasis] =  dsum
                generator_matrix[jbasis,ibasis] = -dsum

       #print(f'\ngenmat {iampld}:')
       #print_matrix(generator_matrix)
       #norm_genmat = np.linalg.norm(generator_matrix)
       #print(f'norm of genmat {iampld}: {norm_genmat}')

        list_genmat.append(generator_matrix)

    return list_genmat

def make_iapair_genmat_new(list_groupped_iapair,list_basis,nparal=1,debug=False):
    """
    Create the generator matrices for Tiiaa_00 operators
    for a multi-electronic basis set
    """
    if debug:
        print('\nIn make_iapair_genmat_new')
        print(f'Memory of list_basis: {float(sys.getsizeof(list_basis))/float(1024**3)} GB')
    list_genmat = []
    n_basis = len(list_basis)
    if debug:
        print(f'n_basis = {n_basis}')
    for grouped_pair in list_groupped_iapair:
        tic = time.perf_counter()
       #print(iampld,list_iapair[iampld])
       #ex_op = FermionOperator()
        # Normalise flat [dmo, vmo] → [[dmo, vmo]] so the len-based dispatch below works.
        if grouped_pair and not isinstance(grouped_pair[0], (list, tuple)):
            grouped_pair = [grouped_pair]
        if len(grouped_pair) == 1:
            [i,a] = grouped_pair[0]
            Tiiaa_00 = get_Tiiaa_00(i,a)
            Top = Tiiaa_00
        elif len(grouped_pair) == 2:
            if not bool(set(grouped_pair[0]) & set(grouped_pair[1])): #[[5,7],[6,8]]-type group
                [[i,a],[j,b]] = grouped_pair
                Top = get_Tiiaa_00(i,a) + get_Tiiaa_00(j,b)
            else:
                [ia_pair,ib_pair] = grouped_pair
                if ia_pair[0] == ib_pair[0] and ia_pair[1] != ib_pair[1]:
                    orb_nondegen = ia_pair[0]
                    orb_degen1 = ia_pair[1]
                    orb_degen2 = ib_pair[1]
                    l_change_sign = False
                elif ia_pair[0] != ib_pair[0] and ia_pair[1] == ib_pair[1]:
                    orb_nondegen = ia_pair[1]
                    orb_degen1 = ia_pair[0]
                    orb_degen2 = ib_pair[0]
                    l_change_sign = True
                else:
                    print(f'The not [[i,a],[i,b]] or [[i,a],[j,a]] pairs detected: {ia_pair,ib_pair}')
                    sys.exit()

                Top = get_sqrt_half_Tiiaa_00_plus_Tiibb_00(orb_nondegen,orb_degen1,orb_degen2)
                if l_change_sign: Top = -Top


        if debug:
            print('\nConstructing generator_matrix for debugging')
            print(f'The genmat will take memory {n_basis * n_basis * 8.0 / 1024 / 1024 / 1024} GB')
            generator_matrix = np.zeros([n_basis,n_basis])
           #Replaced the serial code below by the parallel code. This serial code
           #is for debugging comparison only.
            for ibasis,basis_bra in enumerate(list_basis):
                for jbasis in range(ibasis+1,n_basis):
                    basis_ket = list_basis[jbasis]
                    Tket = op_action_tz_CSF(Top,basis_ket)
                    Selm = overlap_CSFs(basis_bra,Tket)
                    generator_matrix[ibasis,jbasis] =  Selm
                    generator_matrix[jbasis,ibasis] = -Selm
        tic_1 = time.perf_counter()
        i_upper = np.triu_indices(n_basis)
        toc_1 = time.perf_counter()
        if debug:
            print(f'Time to prepare i_upper: {toc_1-tic_1}')
            print(f'Memory of i_upper: {float(sys.getsizeof(i_upper))/float(1024**3)} GB')
            print(f'Dimension of i_upper: {len(i_upper[0]),len(i_upper[1])}')
        tic_1= toc_1
       #i_upper_list = []
       #for i in range(len(i_upper[0])):
       #    i_upper_list.append([i_upper[0][i],i_upper[1][i]])
       #toc_1 = time.perf_counter()
       #print(f'Time to prepare i_upper_list: {toc_1-tic_1}')
       #print(f'Memory of i_upper_list: {float(sys.getsizeof(i_upper_list))/float(1024**3)} GB')
       #tic_1= toc_1

        n_all_pairs = len(i_upper[0])
        index_chunks = even_distrib_ncomp_to_nparal(n_all_pairs,nparal,l_only_end_points=True)
        nchunk = len(index_chunks)
        list_i_upper_chunk = []
        list_list_basis = []
        n_sum_dim = 0
        for ichunk in range(nchunk):
            [istart, iend] = index_chunks[ichunk]
            i_upper_chunk = (i_upper[0][istart:iend+1],i_upper[1][istart:iend+1])
            n_sum_dim += len(i_upper_chunk[0])
            list_i_upper_chunk.append(i_upper_chunk)
            list_list_basis.append(list_basis)

       #Check dimension consistence
        assert len(i_upper[0]) == n_sum_dim

       #chunks = []
       #for ii in range(nchunk):
       #    list_list_basis.append(list_basis)
       #    chunks.append(i_upper_list[index_chunks[ii][0]:index_chunks[ii][-1]+1])
       #    print(f'ii {ii}, Memory of list_list_basis: {float(sys.getsizeof(list_list_basis))/float(1024**3)} GB')

       #toc_1 = time.perf_counter()
       #print(f'Time to prepare chunks: {toc_1-tic_1}')
       #tic_1= toc_1

        def genmat_for_one_list_pairs_braket(i_upper_chunk,list_basis):
            list_output = []
            genmat = lil_matrix((len(list_basis),len(list_basis)))
            for iibra,ibra in enumerate(i_upper_chunk[0]):
                iket = i_upper_chunk[1][iibra]
                basis_bra, basis_ket = list_basis[ibra], list_basis[iket]
               #Jump to next if the bra and ket differ by more than two pairs of dmos
                set_dmo_bra = set(dmo_in_SD(basis_bra[0][0]))
                set_dmo_ket = set(dmo_in_SD(basis_ket[0][0]))
                set_dmo_diff = set_dmo_bra^set_dmo_ket
                if len(set_dmo_diff) > 4:
                    list_output.append(0.0)
                    continue
                Tket = op_action_tz_CSF(Top,basis_ket)
                Selm = overlap_CSFs(basis_bra,Tket)
                list_output.append(Selm)
                genmat[ibra,iket] =  Selm
                genmat[iket,ibra] = -Selm

            genmat = csr_matrix(genmat)
            return genmat

        toc = time.perf_counter()
        tic = toc
       #list_list_outcome = Parallel(n_jobs=nchunk)(delayed(genmat_for_one_list_pairs_braket)(chunks[ii],list_list_basis[ii]) for ii in range(nchunk))
        list_chunk_genmat = Parallel(n_jobs=nchunk)(delayed(genmat_for_one_list_pairs_braket)(list_i_upper_chunk[ii],list_list_basis[ii]) for ii in range(nchunk))
        toc = time.perf_counter()
        if debug:
            print(f'Time consumed by parallel: {toc - tic}')
            print(f'Time consumed before reaching parallel: {toc - tic}')
        tic = toc
       #list_list_outcome = Parallel(n_jobs=nchunk)(delayed(genmat_for_one_list_pairs_braket)(chunks[ii],list_basis) for ii in range(nchunk))
       #genmat_paral = np.zeros([n_basis,n_basis])
        genmat_sparse = csr_matrix((n_basis,n_basis))
       #print(f'genmat_paral array size: {n_basis*n_basis}, takes memory: {genmat_paral.size*genmat_paral.itemsize} bytes')
       #for ichunk, chunk in enumerate(chunks):
       #    print(f'Handling chunk {ichunk}')
       #    list_outcome = list_list_outcome[ichunk]
       #    for ii, ind_braket in enumerate(chunk):
       #        genmat_paral[ind_braket[0],ind_braket[1]] = list_outcome[ii]
       #        genmat_sparse[ind_braket[0],ind_braket[1]] =  list_outcome[ii]
       #        genmat_sparse[ind_braket[1],ind_braket[0]] = -list_outcome[ii]
        for genmat in list_chunk_genmat: genmat_sparse += genmat

        memory_usage_bytes = (
            genmat_sparse.data.nbytes +
            genmat_sparse.indices.nbytes +
            genmat_sparse.indptr.nbytes
        )
       #print(f'Memory taken by genmat_sparse: {memory_usage_bytes / (1024**3)} MB')
       #i_lower = np.tril_indices(n_basis,-1)
       #genmat_paral[i_lower] = -genmat_paral.T[i_lower]
       #toc = time.perf_counter()
       #print(f'Time consumed by making genmat_paral: {toc - tic}')
       #tic = toc
       #if debug: assert np.isclose(np.linalg.norm(genmat_paral - generator_matrix),0.0)
        if debug: assert np.isclose(scipy.sparse.linalg.norm(genmat_sparse - csr_matrix(generator_matrix)),0.0)

       #print(f'\ngenmat {iampld}:')
       #print_matrix(generator_matrix)
       #norm_genmat = np.linalg.norm(generator_matrix)
       #print(f'norm of genmat {iampld}: {norm_genmat}')

       #list_genmat.append(generator_matrix)
        list_genmat.append(genmat_sparse)

    return list_genmat

def make_pair_ex_space_manual(refCSF, grouped_mp2_ia_pair, list_orb_exclude=[], nparal=1,debug=False, build_genmat=True, combo_order=None):
    """
    Build pair excitation space from a reference CSF.

    Each element of grouped_mp2_ia_pair is either:
      [i, a]                                  – regular single ia-pair
      [(c1, [i1, a1]), (c2, [i2, a2]), ...]   – sym-adapted combination
                                                 (list of (coef, [i,a]) tuples)

    Example:
      Regular:    grouped_mp2_ia_pair = [[4,7], [5,8], [0,9]]
      Sym-adapted: grouped_mp2_ia_pair = [(1/√2,[4,7]),(1/√2,[5,8])], [0,9]]
        → slot 1: (G_47+G_58)/√2 |ref>
        → slot 2: G_09|ref>
        → slot 3: G_09·(G_47+G_58)/√2 |ref>   ← cross-term, automatically sym-adapted

    Cross-terms are generated by applying each ordered subset of effective pairs to
    the reference in sequence. Duplicate states (same SD index set) are skipped.
    """
    from itertools import combinations as _iter_comb

    if debug:
        print('\nIn make_pair_ex_space_manual')

    [list_ref_SD, list_ref_idx, ref_coef] = refCSF

    # Occupancy consistency check on the reference
    n_spatialmo = len(list_ref_SD[0]) // 2
    on_spatialmo = list_ref_SD[0][0:2*n_spatialmo:2] + list_ref_SD[0][1:2*n_spatialmo:2]
    if debug: print(on_spatialmo)
    for on in list_ref_SD:
        if not np.allclose(on_spatialmo, on[0:2*n_spatialmo:2] + on[1:2*n_spatialmo:2]):
            print('Inconsistent occupancies of spatial orbitals in a state')
            print(on_spatialmo)
            print(on[0:2*n_spatialmo:2] + on[1:2*n_spatialmo:2])
            print('bombing out')
            sys.exit()

    def _is_sym(p):
        """True when p is a sym-adapted pair [(c,[i,a]), ...]."""
        return len(p) > 0 and isinstance(p[0], tuple)

    def _apply_single(sds, coefs, dmo, vmo):
        """Apply one DOMO→VMO excitation to each SD in sds.
        Returns (new_sds, new_coefs), skipping SDs where the excitation is invalid.
        Returns (None, None) if no SD can undergo the excitation."""
        dmoa, dmob = 2*dmo, 2*dmo+1
        vmoa, vmob = 2*vmo, 2*vmo+1
        out_sds, out_coefs = [], []
        for ii, onvec in enumerate(sds):
            l_fwd = (onvec[dmoa]==1.0 and onvec[dmob]==1.0 and
                     onvec[vmoa]==0.0 and onvec[vmob]==0.0)
            l_rev = (onvec[dmoa]==0.0 and onvec[dmob]==0.0 and
                     onvec[vmoa]==1.0 and onvec[vmob]==1.0)
            if not l_fwd and not l_rev:
                continue   # this SD can't undergo the excitation — skip it
            new_ov = np.array(onvec, dtype=float)
            sign = 1.0
            if l_fwd:
                new_ov[dmoa], new_ov[dmob], new_ov[vmoa], new_ov[vmob] = 0, 0, 1, 1
            else:
                new_ov[vmoa], new_ov[vmob], new_ov[dmoa], new_ov[dmob] = 0, 0, 1, 1
                sign = -1.0
            out_sds.append(new_ov)
            out_coefs.append(float(coefs[ii]) * sign)
        if not out_sds:
            return None, None
        return out_sds, np.array(out_coefs)

    def _apply_group(sds, coefs, group):
        """Apply a sequence of [dmo,vmo] pairs one-by-one.
        group = [[dmo,vmo], ...] — each sub-pair is applied in order."""
        cur_sds, cur_coefs = sds, coefs
        for pair in group:
            cur_sds, cur_coefs = _apply_single(cur_sds, cur_coefs, pair[0], pair[1])
            if cur_sds is None:
                return None, None
        return cur_sds, cur_coefs

    def _apply_eff(sds, coefs, eff_pair):
        """Apply one effective pair to a multi-SD state.  Three formats accepted:

        1. Sym-adapted : [(c1, group1), (c2, group2), ...]
                         where each group = [[dmo,vmo], ...]
           → weighted superposition of each group applied to state

        2. Group format: [[dmo,vmo], [dmo,vmo], ...]   (original caller format)
           → apply each sub-pair sequentially

        3. Flat format : [dmo, vmo]                     (ints)
           → single direct excitation
        """
        if _is_sym(eff_pair):
            # sym-adapted combination: each component is (coef, group)
            all_sds, all_coefs = [], []
            for (c_k, ia_group) in eff_pair:
                new_sds, new_coefs = _apply_group(sds, coefs, ia_group)
                if new_sds is not None:
                    all_sds.extend(new_sds)
                    all_coefs.extend((new_coefs * float(c_k)).tolist())
            if not all_sds:
                return None, None
            return all_sds, np.array(all_coefs)
        elif isinstance(eff_pair[0], list):
            # group format: [[dmo,vmo], ...]
            return _apply_group(sds, coefs, eff_pair)
        else:
            # flat format: [dmo, vmo]
            return _apply_single(sds, coefs, eff_pair[0], eff_pair[1])

    def _canonicalize(sds, coefs, tol=1e-12):
        """Combine duplicate SDs (same on-vector index) by summing their coefficients.
        Drop entries whose combined coefficient is ~0.
        Returns (new_sds, new_idx_list, new_coefs), or (None, None, None) if everything cancels.

        This is necessary for sym-adapted cross-terms: applying two sym-adapted generators
        in sequence can produce the same SD from multiple paths (e.g. G_ab then G_cd and
        G_cd then G_ab via different linear combinations), leading to duplicate rows that
        must be merged before storing the state."""
        from collections import defaultdict
        merged_coef = defaultdict(float)
        sd_by_idx   = {}
        for ov, c in zip(sds, coefs):
            ix = get_on_idx(np.array(ov))
            merged_coef[ix] += float(c)
            sd_by_idx[ix] = ov   # all duplicates are identical bitstrings; keep any one
        kept = [(ix, merged_coef[ix]) for ix in merged_coef if abs(merged_coef[ix]) > tol]
        if not kept:
            return None, None, None   # full cancellation
        kept.sort(key=lambda x: x[0])   # canonical ordering by index
        new_idx   = [k[0] for k in kept]
        new_sds   = [sd_by_idx[k[0]] for k in kept]
        new_coefs = np.array([k[1] for k in kept])
        # Renormalize: path-interference amplitudes from sym-adapted generators accumulate
        # factors of 1/√2 per step, leaving the merged state with |coefs|² < 1.
        # The physical state is the normalised version; scale back to unit norm.
        norm = np.sqrt(np.dot(new_coefs, new_coefs))
        if norm < tol:
            return None, None, None
        new_coefs = new_coefs / norm
        return new_sds, new_idx, new_coefs

    # ── build ex_space ───────────────────────────────────────────────────────

    tic = time.perf_counter()
    list_ex_space = [refCSF]
    seen = {frozenset(list_ref_idx)}   # deduplication by SD index set

    n_eff = len(grouped_mp2_ia_pair)
    # Cap the chained-excitation order at combo_order, consistent with the rest
    # of the pipeline (selection only builds chains up to combo_order). Without a
    # cap this walks the full 2^n_eff power set of pair subsets; with combo_order=k
    # it walks only orders 1..k (~n_eff^k states). combo_order=None preserves the
    # original full-order behavior for other callers.
    max_order = n_eff if combo_order is None else max(1, min(combo_order, n_eff))
    for nex in range(1, max_order + 1):
        for combo in _iter_comb(range(n_eff), nex):
            # Fresh independent copies per combo. copy.deepcopy on a list of
            # numpy arrays is extremely slow and is called 2^n_eff times here;
            # np.array(...) makes the same independent float copies far faster.
            sds   = [np.array(sd, dtype=float) for sd in list_ref_SD]
            coefs = np.array(ref_coef, dtype=float)
            valid = True
            for idx in combo:
                new_sds, new_coefs = _apply_eff(sds, coefs, grouped_mp2_ia_pair[idx])
                if new_sds is None:
                    valid = False
                    break
                sds, coefs = new_sds, new_coefs
            if not valid:
                continue
            # Merge duplicate SDs produced by sym-adapted cross-terms (same bitstring from
            # different application paths → sum their coefficients; drop if they cancel).
            sds, idx_list, coefs = _canonicalize(sds, coefs)
            if sds is None:
                continue   # full cancellation — paths undid each other
            key = frozenset(idx_list)
            if key in seen:
                continue   # duplicate state from a different application ordering
            seen.add(key)
            list_ex_space.append([sds, idx_list, coefs])
            if debug:
                print(f'  combo {combo}: coefs={np.round(coefs, 4)}')
                for sd in sds:
                    print(f'    SD={np.array(sd, dtype=int).tolist()}')

    toc = time.perf_counter()
    if debug:
        print(f'make_pair_ex_space_manual: {len(list_ex_space)} states, '
          f'build took {toc-tic:.3f}s')

    if build_genmat:
        tic = toc
        # Build one combined sparse genmat per effective pair.
        # For sym-adapted pairs [(c1,p1),(c2,p2),...] the combined generator is
        #   G_eff = c1*G_p1 + c2*G_p2 + ...
        # so it correctly encodes both the sym/antisym sign and the 1/√2 normalisation.
        # We build individual sub-genmats first (one parallel call), then sum with coefficients.

        sub_pairs = []   # group-format ia pairs fed to make_iapair_genmat_new
        sub_map   = []   # (eff_idx, coef) for each sub_pairs entry
        for eff_idx, p in enumerate(grouped_mp2_ia_pair):
            if _is_sym(p):
                for (c_k, ia_k) in p:
                    sub_pairs.append(ia_k)          # ia_k is already [[i,a]] group format
                    sub_map.append((eff_idx, float(c_k)))
            else:
                sub_pairs.append(p)
                sub_map.append((eff_idx, 1.0))

        sub_genmats = make_iapair_genmat_new(sub_pairs, list_ex_space, nparal, debug=False)

        # Combine: sum c_k * G_k for each effective pair index
        n_basis = len(list_ex_space)
        combined = [None] * len(grouped_mp2_ia_pair)
        for k, (eff_idx, coef) in enumerate(sub_map):
            g = sub_genmats[k] * coef
            combined[eff_idx] = g if combined[eff_idx] is None else combined[eff_idx] + g

        list_genmat_new = [
            g if g is not None else csr_matrix((n_basis, n_basis))
            for g in combined
        ]
        toc = time.perf_counter()
        if debug:
            print(f'Time to prepare list_genmat_new: {toc - tic:.3f}s')
    else:
        list_genmat_new = []

    return list_ex_space, list_genmat_new

   #The whole subroutine stops here

def make_pair_ex_space_manual_symadapted(refCSF, grouped_mp2_ia_pair, list_orb_exclude=[], nparal=1, build_genmat=True, debug=False):
    """
    Sym-adapted version of make_pair_ex_space_manual.

    Accepts references with MIXED spatial occupancy — e.g. a sym-adapted state
    like (G1+G2)/√2|ref> whose SDs come from two different occupation patterns.
    The original function bombs out for such references because it requires all
    SDs to share the same spatial occupancy.

    Example
    -------
    ref   = (G1+G2)/√2|ref>  with SDs:
              SD_a:  DMOs {0,1,2,3,4,6}   (G1|ref>)
              SD_b:  DMOs {0,1,2,3,5,6}   (G2|ref>)
    ia-pair = G3 = (dmo=7, vmo=9)

    Original function: bombs — SD_a and SD_b have different occupancy.

    This function: applies G3 to EACH SD independently.
      SD_a: dmo=7 not doubly occupied → skip this SD (contributes 0)
      SD_b: dmo=7 doubly occupied     → SD_b excited to SD_b'
    Result slot 1 = [SD_b'], coef = (original coef of SD_b) / norm

    Key differences from make_pair_ex_space_manual:
      1. No occupancy-consistency check.
      2. No single-SD reference discovery via prepare_UCSF_one_CSF_manual.
         Instead, each grouped_pair is tried directly against every SD.
      3. SDs for which a pair-excitation is inapplicable are simply dropped
         from that excitation slot (they contribute zero amplitude).
      4. Cross-term combinations (nex > 1) are handled by iterating over
         combinations of grouped pairs and applying each combination SD-by-SD.
    """
    if debug: print('\nIn make_pair_ex_space_manual_symadapted')

    [list_ref_SD, list_ref_idx, ref_coef] = refCSF
    n_spatialmo = len(list_ref_SD[0]) // 2
    # No occupancy check — mixed occupancy is allowed.

    # Flatten to individual (dmo, vmo) pair groups for combination enumeration.
    # Each element of grouped_mp2_ia_pair is itself a list of (dmo,vmo) pairs
    # that must ALL be applied together as one excitation step.
    from itertools import combinations as _combinations

    list_ex_space = [refCSF]   # slot 0 = reference

    # Try every non-empty subset of grouped_mp2_ia_pair at every excitation level,
    # mirroring the nex loop in the original function.
    n_groups = len(grouped_mp2_ia_pair)
    for nex in range(1, n_groups + 1):
        for group_combo in _combinations(range(n_groups), nex):
            # Flatten the orbital pairs for this combination of groups
            pair_comb = []
            for gi in group_combo:
                for pair in grouped_mp2_ia_pair[gi]:
                    pair_comb.append(pair)
            if any(p[0] in list_orb_exclude or p[1] in list_orb_exclude for p in pair_comb):
                continue

            # Apply pair_comb to each SD in the reference independently.
            new_SDs  = []
            new_idxs = []
            new_coef = []
            for ii, onvec in enumerate(list_ref_SD):
                onvec_ex = np.array(onvec, dtype=float)   # mutable numpy copy
                c_ex     = float(ref_coef[ii])
                l_skip   = False
                for pair in pair_comb:
                    dmo, vmo     = pair[0], pair[1]
                    dmoa, dmob   = 2*dmo,   2*dmo+1
                    vmoa, vmob   = 2*vmo,   2*vmo+1
                    l_d2v0 = (onvec_ex[dmoa]==1.0 and onvec_ex[dmob]==1.0
                              and onvec_ex[vmoa]==0.0 and onvec_ex[vmob]==0.0)
                    l_d0v2 = (onvec_ex[dmoa]==0.0 and onvec_ex[dmob]==0.0
                              and onvec_ex[vmoa]==1.0 and onvec_ex[vmob]==1.0)
                    if l_d2v0:
                        onvec_ex[dmoa]=0.0; onvec_ex[dmob]=0.0
                        onvec_ex[vmoa]=1.0; onvec_ex[vmob]=1.0
                    elif l_d0v2:
                        onvec_ex[vmoa]=0.0; onvec_ex[vmob]=0.0
                        onvec_ex[dmoa]=1.0; onvec_ex[dmob]=1.0
                        c_ex *= -1.0
                    else:
                        l_skip = True   # this SD doesn't support excitation — contributes 0
                        break
                if not l_skip:
                    new_SDs.append(onvec_ex)
                    new_idxs.append(get_on_idx(onvec_ex))
                    new_coef.append(c_ex)

            if len(new_SDs) == 0:
                if debug: print(f'  pair_comb {pair_comb}: no SDs support this excitation, skipping')
                continue

            # Normalise
            coef_arr = np.array(new_coef)
            nrm = np.linalg.norm(coef_arr)
            if nrm < 1e-12:
                continue
            list_ex_space.append([new_SDs, new_idxs, coef_arr / nrm])
            if debug: print(f'  pair_comb {pair_comb}: added slot {len(list_ex_space)-1}')

    if build_genmat:
        list_genmat = make_iapair_genmat_new(grouped_mp2_ia_pair, list_ex_space, nparal, debug=False)
    else:
        list_genmat = []
    return list_ex_space, list_genmat


def make_pair_ex_space_group_manual(refCSF,grouped_mp2_ia_pair, extra_basis_CSF=[], list_orb_exclude = [],nparal=1,debug=False):

    if debug: print('\nIn make_pair_ex_space_manual')

    [list_ref_SD, list_ref_idx, ref_coef] = refCSF

    n_spatialmo = len(list_ref_SD[0]) // 2
    on_spatialmo = list_ref_SD[0][0:2*n_spatialmo:2] + list_ref_SD[0][1:2*n_spatialmo:2]
    if debug: print(on_spatialmo)
    for on in list_ref_SD:
        if not np.allclose(on_spatialmo, on[0:2*n_spatialmo:2] + on[1:2*n_spatialmo:2]):
            print(f'Inconsistent occupancies of spatial orbitals in a state')
            print(on_spatialmo)
            print(on[0:2*n_spatialmo:2] + on[1:2*n_spatialmo:2])
            print('bombing out')
            sys.exit()

    list_somo = list(np.where(on_spatialmo == 1.0)[0])
    list_ref_dmo = list(np.where(on_spatialmo == 2.0)[0])
    list_ref_vmo = list(np.where(on_spatialmo == 0.0)[0])
    if debug:
        print(f'SOMO whose occupancies remain in excitations: {list_somo}')
        print(f'DMO in reference that will be excited from: {list_ref_dmo}')
        print(f'VMO in reference that will be excited from: {list_ref_vmo}')

    refCSF_1SD = [list_ref_SD[0:1],list_ref_idx[0:1],np.array([1.0])]
    if debug:
        print(f'ref of 1SD:')
        print(refCSF_1SD)

    list_theta = []
    import random
    for grouped_pair in grouped_mp2_ia_pair:
        list_theta.append(random.uniform(0.1,0.2))

    UCSF = prepare_UCSF_one_CSF_manual(refCSF_1SD,grouped_mp2_ia_pair,list_theta,nparal)

    if debug:
        print(f'UCSF that contains all possible pair excitations:')
        print(UCSF)

    tic = time.perf_counter()
    list_all_ex = []
    for iexSD, exSD in enumerate(UCSF[0][1:]):
        on_spatialmo_exSD = exSD[0:2*n_spatialmo:2] + exSD[1:2*n_spatialmo:2]
        list_dmo = list(np.where(on_spatialmo_exSD == 2.0)[0])
        list_vmo = list(np.where(on_spatialmo_exSD == 0.0)[0])
        if debug:
            print(f'exSD {iexSD}, list_dmo: {list_dmo}, list_vmo: {list_vmo}')

        set_hole_orb = list(set(list_ref_dmo) & set(list_vmo))
        set_part_orb = list(set(list_ref_vmo) & set(list_dmo))
        assert len(set_hole_orb) == len(set_part_orb)
        if len(set_hole_orb) == 0:
            print(f'The reference SD is not the 0th one. Bombing out!')
            sys.exit()
       #if debug:
       #    print(f'exSD {iexSD}, hole_orb: {set_hole_orb}, part_orb: {set_part_orb}')

        list_pair_ex = []
        for ii, hole in enumerate(set_hole_orb):
            list_pair_ex.append([hole,set_part_orb[ii]])
        if debug: print(f'list_pair_ex: {list_pair_ex}')
        list_all_ex.append(list_pair_ex)

    if debug:
        print(f'All pair excitations:')
        for item in list_all_ex:
            print(item)

    toc = time.perf_counter()
    print(f'Time to prepare list_all_ex: {toc - tic}')
    tic = toc

    max_nex = 0
    for item in list_all_ex:
        if len(item) > max_nex: max_nex = len(item)

    if debug: print(f'Maximum pair excitaiton level: {max_nex}')

    list_collected_ex = []
    for nex in range(1,max_nex+1):
        list_ex_one_ex_level = []
        for item in list_all_ex:
            if len(item) == nex: list_ex_one_ex_level.append(item)
       #print(f'list_ex_one_ex_level for nex {nex}: {list_ex_one_ex_level}')

        list_collected_ex.append(list_ex_one_ex_level)

    ndim_pair_ex = 0
    for nex in range(1,max_nex+1):
        if debug:
            print(f'Combinations of {nex} dmo to {nex} vmo mp2 excitations:')
            print(list_collected_ex[nex-1])
        ndim_pair_ex += len(list_collected_ex[nex-1])

    if debug: print(f'\nTotal # of multiple pair excitations: {ndim_pair_ex}')
    toc = time.perf_counter()
    print(f'Time to prepare list_collected_ex: {toc - tic}')
    tic = toc

    list_ex_space = [refCSF]
    for nex in range(1,max_nex+1):
        list_one_ex_level = list_collected_ex[nex-1]
        for pair_comb in list_one_ex_level:
            l_skip = False
            if debug: print(f'pair_comb: {pair_comb}')
            list_ex_SD = copy.deepcopy(list_ref_SD)
            list_ex_coef = copy.deepcopy(ref_coef)
            for pair in pair_comb:
                dmo, vmo = pair[0],pair[1]
                dmoa, dmob,vmoa, vmob = 2*dmo, 2*dmo+1, 2*vmo, 2*vmo+1
                for ii, onvec in enumerate(list_ex_SD):
                   #judge whether dmo is doubly occupied and vmo is empty, or
                   #whether dmo is empty and vmo is doubly occupied. All the other situations are not good!
                    l_ndmo2_nvmo0,  l_ndmo0_nvmo2 = False, False
                    if onvec[dmoa] == 1.0 and onvec[dmob] == 1.0 and onvec[vmoa] == 0.0 and onvec[vmob] == 0.0:
                        l_ndmo2_nvmo0 = True
                    if onvec[dmoa] == 0.0 and onvec[dmob] == 0.0 and onvec[vmoa] == 1.0 and onvec[vmob] == 1.0:
                        l_ndmo0_nvmo2 = True
                    if l_ndmo2_nvmo0 and l_ndmo0_nvmo2:
                        print(f'l_ndmo2_nvmo0 and l_ndmo0_nvmo2 shall not be both True')
                        print(onvec,dmo,vmo)
                        print('Bombing out')
                        sys.exit()
                    elif not l_ndmo2_nvmo0 and not l_ndmo0_nvmo2:
                        print(f'\nAll ia pair combinations at this level: {list_one_ex_level}')
                        print(f'This pair combination {pair_comb}')
                        print(f'MO {dmo} is not doubly occupied or empty, or MO {vmo} is not doubly occupied or empty')
                        print(onvec)
                        print(f'list_ref_SD:')
                        print(list_ref_SD)
                        print(f'{dmo} and/or {vmo} shall be in list_orb_exclude:?')
                        print(list_orb_exclude)
                       #print('Bombing out!')
                       #sys.exit()
                        print('Skipping this pair combination')
                        l_skip = True
                    elif l_ndmo2_nvmo0:
                        onvec[dmoa],onvec[dmob],onvec[vmoa],onvec[vmob] = 0.0,0.0,1.0,1.0
                    elif l_ndmo0_nvmo2:
                        onvec[vmoa],onvec[vmob],onvec[dmoa],onvec[dmob] = 0.0,0.0,1.0,1.0
                        list_ex_coef *= -1.0

                    if l_skip: break
                if l_skip: break
            list_ex_idx = []
            for onvec in list_ex_SD:
                list_ex_idx.append(get_on_idx(onvec))
            list_ex_space.append([list_ex_SD,list_ex_idx,list_ex_coef])
           #print(f'list_ex_SD: {list_ex_SD}')
    toc = time.perf_counter()
    print(f'Time to prepare list_ex_space: {toc - tic}')
    tic = toc

   #make list_genmat for the excited space
   #list_group_iapair = []
   #for item in mp2_ia_pair:
   #    list_group_iapair.append([item])
   #print(f'list_group_iapair: {list_group_iapair}')
   #tic = time.perf_counter()
   #list_genmat = make_iapair_genmat(grouped_mp2_ia_pair,list_ex_space)
   #toc = time.perf_counter()
   #time_make_iapair_genmat = toc - tic
   #tic = time.perf_counter()
    list_ex_space = list_ex_space + extra_basis_CSF

    list_genmat_new = make_iapair_genmat_new(grouped_mp2_ia_pair, list_ex_space, nparal, debug=False)
    toc = time.perf_counter()
    print(f'Time to prepare list_genmat_new: {toc - tic}')
    tic = toc
   #toc = time.perf_counter()
   #time_make_iapair_genmat_new = toc - tic
   #print(f'Times for old and new making genmats: {time_make_iapair_genmat,time_make_iapair_genmat_new}')
   #assert len(list_genmat) == len(list_genmat_new)
   #for igenmat in range(len(list_genmat)):
   #    assert np.isclose(np.linalg.norm(list_genmat_new[igenmat] - list_genmat[igenmat]),0.0)

   #list_genmat_sparse = []
   #for genmat in list_genmat_new:
   #    list_genmat_sparse.append(csr_matrix(genmat))
   #toc = time.perf_counter()
   #print(f'Time to prepare list_genmat_sparse: {toc - tic}')
   #tic = toc
   #return list_ex_space, list_genmat_sparse
    return list_ex_space, list_genmat_new



def make_pair_ex_space_fast(list_ref_SD,list_ref_idx,list_ref_coef,mp2_ia_pair,list_orb_exclude = [],debug=False):
    """
    list_orb_exclude contains the explicitly specified orbitals that are not included in dmo-to-vmo excitations.
    """

    if debug: print('\nIn make_pair_ex_space_fast')

    n_spatialmo = len(list_ref_SD[0]) // 2
    on_spatialmo = list_ref_SD[0][0:2*n_spatialmo:2] + list_ref_SD[0][1:2*n_spatialmo:2]
    if debug: print(on_spatialmo)
    for on in list_ref_SD:
        if not np.allclose(on_spatialmo, on[0:2*n_spatialmo:2] + on[1:2*n_spatialmo:2]):
            print(f'Inconsistent occupancies of spatial orbitals in a state')
            print(on_spatialmo)
            print(on[0:2*n_spatialmo:2] + on[1:2*n_spatialmo:2])
            print('bombing out')
            sys.exit()

    list_somo = list(np.where(on_spatialmo == 1.0)[0])
    if debug: print(f'SOMO whose occupancies remain in excitations: {list_somo}')


    pair_included = []
   #exclude excitations that involve SOMO
    for ia_pair in mp2_ia_pair:
        l_include = True
        nocc_pair = 0.0
        for orb in ia_pair:
            if orb in list_somo: l_include = False
            if orb in list_orb_exclude: l_include = False
            nocc_pair += nocc_spatial_orb_LCSD(list_ref_SD,list_ref_coef,orb)

        if not np.isclose(nocc_pair,2.0): l_include = False


        if l_include: pair_included.append(ia_pair)



    sorted_pair = sorted(pair_included, key=lambda x: x[0])[::-1]
    if debug:
        print(f'ia_pair included in make_pair_ex_space_fast: {sorted_pair}')


    list_dmo = []
    list_vmo = []
    for ia_pair in sorted_pair:
        dmo, vmo = ia_pair[0], ia_pair[1]
        if dmo not in list_dmo: list_dmo.append(dmo)
        if vmo not in list_vmo: list_vmo.append(vmo)

    max_nex = min(len(list_dmo),len(list_vmo))
    if debug:
        print(f'List of doubly occupied orbitals included: {list_dmo}')
        print(f'List of virtual         orbitals included: {list_vmo}')
        print(f'maximum pair excitation foldness: {max_nex}')

   #Group the ia pairs based on their occupied orbitals
    list_grouped_pairs_by_dmo = []
    for dmo in list_dmo:
        list_pair_1_dmo = []
        for pair in sorted_pair:
            if pair[0] == dmo:
                if debug: print(f'Adding {pair} to list')
                list_pair_1_dmo.append(pair)
        if debug: print(list_pair_1_dmo)
        list_grouped_pairs_by_dmo.append(list_pair_1_dmo)

    if debug: print(list_grouped_pairs_by_dmo)


    list_collected_ex = []
    for nex in range(1,max_nex+1):
        list_ex_one_ex_level = []
        comb_occ = list(combinations(list_dmo,nex))
        comb_vir = list(combinations(list_vmo,nex))
        if debug:
            print(f'Combinations of occupied spatial orbitals:')
            print(comb_occ)
            print(f'Combinations of virtual  spatial orbitals:')
            print(comb_vir)
        for occmo_list in comb_occ:
            if debug: print(f'occmo_list: {occmo_list}')
            list_dmo_group = []
            for dmo in [*occmo_list]:
                idmo_group = np.where(np.array(list_dmo) == dmo)[0][0]
                list_dmo_group.append(idmo_group)
            if debug: print(f'list_dmo_group: {list_dmo_group}')
            list_tmp = []
            for ii, dmo_group in enumerate(list_dmo_group):
                list_tmp.append(list_grouped_pairs_by_dmo[dmo_group])

            if debug: print(f'list_tmp: {list_tmp}')
            list_tmp = list(itertools.product(*list_tmp))
            if debug: print(f'list_tmp: {list_tmp}')
            l_remove_list = []
            for item in list_tmp:
               #If there this set consists of the same set of dmos and vmos as a previous set, remove this set
               #E.g., ([3, 6], [2, 5]) vs ([3, 5], [2, 6])
                list_dmo_tmp = []
                list_vmo_tmp = []
                for pair in item:
                    list_dmo_tmp.append(pair[0])
                    list_vmo_tmp.append(pair[1])
                if debug:
                    print(f'list_dmo_tmp: {list_dmo_tmp}')
                    print(f'list_vmo_tmp: {list_vmo_tmp}')
                l_remove = False
                idx_item = list_tmp.index(item)
                for jtem in list_tmp[:idx_item]:
                    list_dmo_jtem = []
                    list_vmo_jtem = []
                    for pair in jtem:
                        list_dmo_jtem.append(pair[0])
                        list_vmo_jtem.append(pair[1])

                    lsame_dmo, _, _ =two_lists_with_same_contents(list_dmo_jtem,list_dmo_tmp)
                    lsame_vmo, _, _ =two_lists_with_same_contents(list_vmo_jtem,list_vmo_tmp)
                    if lsame_dmo and lsame_vmo:
                        l_remove = True
                        break
                if l_remove:
                    if debug:
                        print('This combination will be removed for containing the same dmos and vmos as a previous set')
                    l_remove_list.append(l_remove)
                    continue
                for dmo in list_dmo_tmp:
                    if len(np.where(np.array(list_dmo_tmp) == dmo)[0]) > 1:
                        l_remove = True
                        break
                for vmo in list_vmo_tmp:
                    if len(np.where(np.array(list_vmo_tmp) == vmo)[0]) > 1:
                        l_remove = True
                        break
                if l_remove:
                    if debug: print('This combination will be removed')
                l_remove_list.append(l_remove)
            if debug: print(f'l_remove_list: {l_remove_list}')
            for ii in range(len(l_remove_list)-1,-1,-1):
                if l_remove_list[ii]: del list_tmp[ii]
            if debug: print(f'list_tmp after removals: {list_tmp}')
            list_tmp = np.array(list_tmp).tolist()
            if debug: print(f'list_tmp after conversion to list of list: {list_tmp}')
            list_ex_one_ex_level.extend(list_tmp)
        list_collected_ex.append(list_ex_one_ex_level)

    ndim_pair_ex = 0
    for nex in range(1,max_nex+1):
        if debug:
            print(f'Combinations of {nex} dmo to {nex} vmo mp2 excitations:')
            print(list_collected_ex[nex-1])
        ndim_pair_ex += len(list_collected_ex[nex-1])

    if debug: print(f'\nTotal # of multiple pair excitations: {ndim_pair_ex}')
    ndim_ex_space = ndim_pair_ex + 1 # +1 to include the reference state as the 0th element in list_ex_space

    list_ex_space = [[list_ref_SD,list_ref_idx,list_ref_coef]]
    for nex in range(1,max_nex+1):
        list_one_ex_level = list_collected_ex[nex-1]
        for pair_comb in list_one_ex_level:
            l_skip = False
            if debug: print(f'pair_comb: {pair_comb}')
            list_ex_SD = copy.deepcopy(list_ref_SD)
            list_ex_coef = copy.deepcopy(list_ref_coef)
           #print(f'list_ex_SD: {list_ex_SD}')
           #print(f'list_ref_SD: {list_ref_SD}')
            for pair in pair_comb:
                dmo, vmo = pair[0],pair[1]
                dmoa, dmob,vmoa, vmob = 2*dmo, 2*dmo+1, 2*vmo, 2*vmo+1
                for ii, onvec in enumerate(list_ex_SD):
                   #judge whether dmo is doubly occupied and vmo is empty, or
                   #whether dmo is empty and vmo is doubly occupied. All the other situations are not good!
                    l_ndmo2_nvmo0,  l_ndmo0_nvmo2 = False, False
                    if onvec[dmoa] == 1.0 and onvec[dmob] == 1.0 and onvec[vmoa] == 0.0 and onvec[vmob] == 0.0:
                        l_ndmo2_nvmo0 = True
                    if onvec[dmoa] == 0.0 and onvec[dmob] == 0.0 and onvec[vmoa] == 1.0 and onvec[vmob] == 1.0:
                        l_ndmo0_nvmo2 = True
                    if l_ndmo2_nvmo0 and l_ndmo0_nvmo2:
                        print(f'l_ndmo2_nvmo0 and l_ndmo0_nvmo2 shall not be both True')
                        print(onvec,dmo,vmo)
                        print('Bombing out')
                        sys.exit()
                    elif not l_ndmo2_nvmo0 and not l_ndmo0_nvmo2:
                        print(f'\nAll ia pair combinations at this level: {list_one_ex_level}')
                        print(f'This pair combination {pair_comb}')
                        print(f'MO {dmo} is not doubly occupied or empty, or MO {vmo} is not doubly occupied or empty')
                        print(onvec)
                        print(f'list_ref_SD:')
                        print(list_ref_SD)
                        print(f'{dmo} and/or {vmo} shall be in list_orb_exclude:?')
                        print(list_orb_exclude)
                       #print('Bombing out!')
                       #sys.exit()
                        print('Skipping this pair combination')
                        l_skip = True
                    elif l_ndmo2_nvmo0:
                        onvec[dmoa],onvec[dmob],onvec[vmoa],onvec[vmob] = 0.0,0.0,1.0,1.0
                    elif l_ndmo0_nvmo2:
                        onvec[vmoa],onvec[vmob],onvec[dmoa],onvec[dmob] = 0.0,0.0,1.0,1.0
                        list_ex_coef *= -1.0

                    if l_skip: break
                if l_skip: break
            list_ex_idx = []
            for onvec in list_ex_SD:
                list_ex_idx.append(get_on_idx(onvec))
            list_ex_space.append([list_ex_SD,list_ex_idx,list_ex_coef])
           #print(f'list_ex_SD: {list_ex_SD}')

   #Check whether the same space is generated by explicit excitations
    if debug:
        icount = 0
        for nex in range(1,max_nex+1):
            list_one_ex_level = list_collected_ex[nex-1]
            for pair_comb in list_one_ex_level:
               #icount += 1
                ex_op = FermionOperator('')
                for pair in pair_comb:
                    dmo, vmo = pair[0],pair[1]
                    E_pair_ia = get_pair_excite(dmo,vmo)
                    D_pair_ia = get_pair_excite(vmo,dmo)
                   #ex_op *= E_pair_ia
                    ex_op *= (E_pair_ia - D_pair_ia)

                on_list_post_ex, idx_list_post_ex,coef_list_post_ex = \
                    op_action_tz_remove_0coef(ex_op,list_ref_SD,list_ref_idx,list_ref_coef)
                if len(on_list_post_ex) == 0: continue
                icount += 1
                list_SD_tmp = list_ex_space[icount][0]
                list_idx_tmp = list_ex_space[icount][1]
                list_coef_tmp = list_ex_space[icount][2]
                lbomb = False
                if len(list_SD_tmp) != len(on_list_post_ex): lbomb = True
                for ii, onvec in enumerate(list_SD_tmp):
                    if not np.isclose(np.linalg.norm(onvec - on_list_post_ex[ii]),0.0): lbomb = True
                    if list_idx_tmp[ii] != idx_list_post_ex[ii]: lbomb = True
                if not np.isclose(np.linalg.norm(list_coef_tmp - coef_list_post_ex),0.0): lbomb = True
                if lbomb:
                    print('\nInconsistent list of SDs from manual excitations and ex_op')
                    print(f'pair combination: {pair_comb}')
                    print(list_SD_tmp)
                    print(on_list_post_ex)
                    print('Bombing out')
                    sys.exit()

    return list_ex_space, list_collected_ex


def make_pair_ex_space(list_ref_SD,list_ref_idx,list_ref_coef,hdmo,lvmo,nmo,debug=False):
    """
    Read in a reference state and return list of states generated by all possible
    combinations of pair excitations from doubly occupied orbitals to virtual
    orbitals. The read-in state consists of a list of SDs and a list of coefficients.
    The read-in list of integer indices are to be used for the action function only.
    They are meaningless at this moment.
    hdmo means the highest doubly occupied orbitals and lvmo means
    the lowest virtual occupied orbitals
    """

    if debug: print('in make_pair_ex_space')

    list_dmo = []
    list_vmo = []
    for i in range(0,hdmo+1):
        list_dmo.append(i)
    for a in range(lvmo,nmo):
        list_vmo.append(a)


    list_ex_space = [[list_ref_SD,list_ref_idx,list_ref_coef]]

    for nex in range(1,min(len(list_dmo),len(list_vmo))+1):
        if debug: print(f'pair excitations involving {nex} occupied spatial orbitals')
        comb_occ = list(combinations(list_dmo,nex))
        comb_vir = list(combinations(list_vmo,nex))

        if debug:
            print(f'Combinations of occupied spatial orbitals:')
            print(comb_occ)
            print(f'Combinations of virtual  spatial orbitals:')
            print(comb_vir)

        for occmo_list in comb_occ:
            for virmo_list in comb_vir:
               #print('occmo_list',[*occmo_list])
               #print('virmo_list',[*virmo_list])

                ex_op = FermionOperator('')
                counter = -1
                for ja,i in enumerate([*occmo_list]):
                    a = [*virmo_list][ja]
                    counter += 1
                    E_pair_ia = get_pair_excite(i,a)
                    ex_op *= E_pair_ia
                   #print(E_pair_ia)
                   #if counter == 0:
                   #    ex_op  = E_pair_ia
                   #   #print(f'ex_op 1st, {ex_op}')
                   #else:
                   #    ex_op *= E_pair_ia
                       #print(f'ex_op, {ex_op}')

                   #ex_op = normal_ordered(ex_op)

               #print(f'ex_op: {ex_op}')
                on_list_post_ex, idx_list_post_ex,coef_list_post_ex = op_action_tz_remove_0coef(ex_op,list_ref_SD,list_ref_idx,list_ref_coef)
                if len(on_list_post_ex) == 0:
                    if debug:
                        print('The excitaiton operator creates a vacuum state')
                        print('occmo_list',[*occmo_list])
                        print('virmo_list',[*virmo_list])
                else:
                    list_ex_space.append([on_list_post_ex,idx_list_post_ex,coef_list_post_ex])

    if debug:
        print(f'List of multiple pair excitation space:')
        for i,item in enumerate(list_ex_space):
            print(item)

    return list_ex_space

def prepare_mp2_amplitudes(homo,orbene,tbt,debug=False):
    """
    Prepare mp2 amplitudes, correlation energies, etc. The read in tbt is for spin orbitals
    and tbt[p,q,r,s]p^+ q^+ r s gives the 2-el term in Hamiltonian. I.e., spin orbitals in
    1st and 4th indices are for one electron, and those in 2nd and 3rd are for the other electron.
    """

    ia_pair_list = []
    mp2_ampld_list = []
    mp2_Ecorr_list = []
    lumo = homo+1
    nmo = len(orbene)

    for i in range(homo+1):
        for a in range(lumo,nmo):
            denominator = orbene[i] - orbene[a]
            numerator = tbt[2*a,2*a,2*i,2*i]
            Ecorr_1pair = (2.0*numerator)**2/(2.0*denominator) #The 2.0 multiplication is to compensate the 1/2 scaling for the 2-el integral.
            if debug:
                print(i,a,orbene[i],orbene[a],denominator,numerator,numerator/denominator,Ecorr_1pair)
            ia_pair_list.append([i,a])
            mp2_ampld_list.append(numerator/denominator)
            mp2_Ecorr_list.append(Ecorr_1pair)

    return mp2_ampld_list,mp2_Ecorr_list,ia_pair_list

def prepare_mp2_amplitudes_actmo(hole_upbound,part_lowbound,orbene,tbt,l_no_sym=False,l_exclud_act_pair=True,debug=False):
    """
    Prepare mp2 amplitudes, correlation energies, etc. The read in tbt is for spin orbitals
    and tbt[p,q,r,s]p^+ q^+ r s gives the 2-el term in Hamiltonian. I.e., spin orbitals in
    1st and 4th indices are for one electron, and those in 2nd and 3rd are for the other electron.
    """

    ia_pair_list = []
    mp2_ampld_list = []
    mp2_Ecorr_list = []
    nmo = len(orbene)

    for i in range(hole_upbound+1):
        for a in range(part_lowbound,nmo):
           #if debug: print(f'main loop in prepare_mp2_amplitudes_actmo: {i,a}')
            if i >= a:
               #print(f'next ia pair since i {i} >= a {a}')
                continue
           #if i >= part_lowbound and a <= hole_upbound:
           #    print(f'next iapair since i {i} >= part_lowbound {part_lowbound} and a {a} <= hole_upbound {hole_upbound}')
           #    continue
            denominator = orbene[i] - orbene[a]
            numerator = tbt[2*a,2*a,2*i,2*i]
            if debug:
                print(i,a,orbene[i],orbene[a],denominator,numerator,numerator/denominator)
            if np.isclose(denominator,0.0):
                if l_no_sym:
                    mp2_ampld_list.append(0.25*np.pi)
                    mp2_Ecorr_list.append(-1.0) # -1.0 is a trivial value added to avoide singularity
                    print(f'Hard-coded values are assigned for degenerate {i,a}, as the degenerate CSF is created by U')
                    ia_pair_list.append([i,a])
                else:
                    print(f'Skip degenerate orbital pairs as the degenerate CSFs are generated by create_missing_axial_sym_CSFs, instead of U')
                continue
            ampld = numerator/denominator
            ia_pair_list.append([i,a])
            Ecorr_1pair = (2.0*numerator)**2/(2.0*denominator) #The 2.0 multiplication is to compensate the 1/2 scaling for the 2-el integral.
           #if abs(ampld) > 0.25*np.pi:
           #    ampld = np.sign(ampld)*0.25*np.pi
           #    Ecorr_1pair = -1.0
            mp2_ampld_list.append(ampld)
            mp2_Ecorr_list.append(Ecorr_1pair)

    assert len(mp2_ampld_list) == len(mp2_Ecorr_list)
    assert len(mp2_ampld_list) == len(ia_pair_list)

    return mp2_ampld_list,mp2_Ecorr_list,ia_pair_list

def sort_mp2_amplitudes(mp2_ampld_list,mp2_Ecorr_list,ia_pair_list,debug=False):
    """
    Sort the mp2 amplitudes and correspondingly reorder the list of mp2 correlation
    energies and i a orbital pairs
    """

   #zipped_list = zip(mp2_ampld_list,ia_pair_list,mp2_Ecorr_list)
   #if debug:  print(list(zipped_list))
    sorted_ampld_list, sorted_ia_pair_list,sorted_Ecorr_list  = zip(*sorted(zip(mp2_ampld_list,ia_pair_list,mp2_Ecorr_list)))
   #sorted_ampld_list, sorted_ia_pair_list,sorted_Ecorr_list = zip(*zipped_list)
    sorted_ampld_list = list(sorted_ampld_list)
    sorted_ia_pair_list = list(sorted_ia_pair_list)
    sorted_Ecorr_list = list(sorted_Ecorr_list)
    if debug:
        print('Sorted mp2 amplitudes')
        print(sorted_ampld_list)
        print('Correspondingly reordered orbital pairs')
        print(sorted_ia_pair_list)
        print('Correspondingly reordered mp2 correlation energies')
        print(sorted_Ecorr_list)

    return sorted_ampld_list, sorted_ia_pair_list, sorted_Ecorr_list

def remove_mp2_amplitudes_old(sorted_ampld_list,sorted_ia_pair_list,sorted_Ecorr_list,small_thrsh,list_act_orb=[]):
    """
    Remove the mp2 excitations whose amplitudes are smaller than the small_thrsh. The read-in
    list of amplitudes must have been sorted
    """

    n_ampld = len(sorted_ampld_list)
    list_removed_pair = []
    for i in range(n_ampld-1,-1,-1):
        [i_orb, a_orb] = sorted_ia_pair_list[i]
        if i_orb in list_act_orb and a_orb in list_act_orb: continue
        if abs(sorted_ampld_list[i]) < small_thrsh:
            list_removed_pair.append(sorted_ia_pair_list[i])
            sorted_ampld_list.pop()
            sorted_ia_pair_list.pop()
            sorted_Ecorr_list.pop()

    print(f'\nRemoved MP2 ia pairs:')
    print(list_removed_pair)

def remove_mp2_amplitudes(sorted_ampld_list,sorted_ia_pair_list,sorted_Ecorr_list,small_thrsh,list_mo_exclud=[],list_act_orb=[],l_remove_actorb_pair=False):
    """
    Remove the mp2 excitations whose amplitudes are smaller than the small_thrsh. When both i and a
    orbitals are in list_act_orb, the excitation is not removed, no matter how small the amplitude is.
    But if l_remove_actorb_pair = True, then the active orbita pairs will be removed.
    """

    n_ampld = len(sorted_ampld_list)
    list_l_remove = [False] * n_ampld
    list_removed_pair = []
    for i in range(n_ampld-1,-1,-1):
        [i_orb, a_orb] = sorted_ia_pair_list[i]
       #if i_orb in list_act_orb and a_orb in list_act_orb and not l_remove_actorb_pair: continue
        if abs(sorted_ampld_list[i]) < small_thrsh or\
          (i_orb in list_act_orb and a_orb in list_act_orb and l_remove_actorb_pair) or \
          (i_orb in list_mo_exclud or a_orb in list_mo_exclud):# or \
         #sorted_ampld_list[i] > 0.0:
            list_l_remove[i] = True
            list_removed_pair.append(sorted_ia_pair_list[i])
           #sorted_ampld_list.pop()
           #sorted_ia_pair_list.pop()
           #sorted_Ecorr_list.pop()
        if (i_orb in list_mo_exclud or a_orb in list_mo_exclud):
            print(f'Removing pair: {sorted_ia_pair_list[i]} for orbital in list_mo_exclud')

    print(f'\nRemoved MP2 ia pairs:')
    print(list_removed_pair)

    for i in range(n_ampld-1,-1,-1):
        if list_l_remove[i]:
            del sorted_ampld_list[i]
            del sorted_ia_pair_list[i]
            del sorted_Ecorr_list[i]

def group_mp2_amplitudes(sorted_ampld_list,sorted_ia_pair_list,sorted_Ecorr_list,l_no_groupping=False):
    """
    Group the sorted mp2 amplitudes. If some of them are degenerate, they are grouped together
    as one item. The orbital pairs and correation energies are also grouped correspondingly.

    """

    group_mp2_ampld = []
    group_mp2_iapair = []
    group_mp2_Ecorr = []
    pair_counted = []

    for i in range(len(sorted_ia_pair_list)):
        if i in pair_counted: continue
        group_mp2_ampld.append(sorted_ampld_list[i])
        pair_list_of_1_ampld = []
        Ecorr_list_of_1_ampld = []
        pair_list_of_1_ampld.append(sorted_ia_pair_list[i])
        Ecorr_list_of_1_ampld.append(sorted_Ecorr_list[i])
        pair_counted.append(i)
        if not l_no_groupping:
            for j in range(i+1,len(sorted_ia_pair_list)):
                if np.isclose(sorted_ampld_list[j],sorted_ampld_list[i]):
                    pair_list_of_1_ampld.append(sorted_ia_pair_list[j])
                    Ecorr_list_of_1_ampld.append(sorted_Ecorr_list[j])
                    pair_counted.append(j)
                else:
                    break

        group_mp2_iapair.append(pair_list_of_1_ampld)
        group_mp2_Ecorr.append(Ecorr_list_of_1_ampld)

    return group_mp2_ampld,group_mp2_iapair,group_mp2_Ecorr

def make_H_matrix_in_pair_ex_space(Hop,list_pair_ex_space,debug=False):
    """
    Read in a real hermitian operator and a list of real states in a multiple pair excitaiton space
    and construct the real symmetric matrix in that space
    """

    ndim = len(list_pair_ex_space)
    if debug: print(f'ndim: {ndim}')

    Hmat = np.zeros([ndim,ndim])
    for ibas in range(ndim):
        basis_bra = list_pair_ex_space[ibas]
       #print(basis_bra)
        SDs_bra = basis_bra[0]
        coefs_bra = basis_bra[2] #basis_bra[1] contains the list of integer indices, not useful here
        for jbas in range(ibas,ndim):
            basis_ket = list_pair_ex_space[jbas]
            SDs_ket   = basis_ket[0]
            coefs_ket = basis_ket[2]

            dsum = 0.0
            for iSD_bra in range(len(SDs_bra)):
                onl = SDs_bra[iSD_bra]
                coefl = coefs_bra[iSD_bra]
                for jSD_ket in range(len(SDs_ket)):
                    onr = SDs_ket[jSD_ket]
                    coefr = coefs_ket[jSD_ket]
                    dsum += coefl*coefr*braket_tz(onl,onr,Hop)

           #print(ibasis,jbasis,dsum)
            Hmat[ibas,jbas]=dsum
            Hmat[jbas,ibas]=dsum

    if debug:
        print(f'Constructed matrix')
        print_matrix(Hmat)

   #Also construct the sparse matrix
    Hmat_sparse = csr_matrix(Hmat)

    return Hmat, Hmat_sparse

def calculate_elm(op,CSFbra,CSFket):
    """
    Calculate matrix elements between two CSFs for a generic Fermionic operator
    """

    [onlist_bra,ind_bra,coefs_bra] = CSFbra
    [onlist_ket,ind_ket,coefs_ket] = CSFket

    dsum = 0.0
    for iSD_bra, onl in enumerate(onlist_bra):
        coefl = coefs_bra[iSD_bra]
        for iSD_ket, onr in enumerate(onlist_ket):
            coefr = coefs_ket[iSD_ket]
            for term in op:
                dsum += coefl*coefr*braket_tz(onl,onr,term)

    return dsum

def make_H_matrix_in_pair_ex_2spaces(Hop,list_pair_ex_space1,list_pair_ex_space2,debug=False):
    """
    Read in a operator and two lists of real states in two multiple pair excitation spaces
    and construct the real a-symmetric matrix between the two spaces. In space1 are bra states and in
    space2 are ket states.
    """

    ndim1 = len(list_pair_ex_space1)
    ndim2 = len(list_pair_ex_space2)
    if debug: print(f'bra dimension: {ndim1}, ket dimension: {ndim2}')

   #if debug:
   #    for term in Hop:
   #        print(term)

    Hmat = np.zeros([ndim1,ndim2])
    for ibas in range(ndim1):
        basis_bra = list_pair_ex_space1[ibas]
        SDs_bra = basis_bra[0]
        coefs_bra = basis_bra[2] #basis_bra[1] contains the list of integer indices, not useful here
        for jbas in range(ndim2):
           #print(f'ibas,jbas {ibas,jbas}')
            basis_ket = list_pair_ex_space2[jbas]
            SDs_ket   = basis_ket[0]
            coefs_ket = basis_ket[2]

            dsum = 0.0
            for iSD_bra in range(len(SDs_bra)):
                onl = SDs_bra[iSD_bra]
                coefl = coefs_bra[iSD_bra]
                for jSD_ket in range(len(SDs_ket)):
                    onr = SDs_ket[jSD_ket]
                    coefr = coefs_ket[jSD_ket]
                    if ibas == 2 and jbas == 1:
                       #print(coefl,onl,coefr,onr,braket_tz(onl,onr,Hop))
                        for term in Hop:
                            term_wise_element = braket_tz(onl,onr,term)
                            if abs(term_wise_element) > 1.e-5 and debug:
                                print(term,onl,onr,term_wise_element,coefl,coefr)
                    dsum += coefl*coefr*braket_tz(onl,onr,Hop)
                   #print(f'dsum = {dsum}')

            Hmat[ibas,jbas] = dsum

    if debug:
        print(f'Constructed matrix in make_H_matrix_in_pair_ex_2spaces')
        print_matrix(Hmat)

    return Hmat

def make_short_H_ferm_op(const,obt_phys,tbt_phys):
    """
    Read in a hermitian fermionic operator and change it from sum p q r s
    to sum p>q, r>s for the 2-body terms
    """

   #print(f'In clean_H_ferm_op')

   #print(f'const: {const}')

    N = obt_phys.shape[0]
   #print(f'# of spin orbitals: {N}')

    H1 = FermionOperator()
    H2 = FermionOperator()
    for p in range(N):
        for q in range(p,N):
            if not np.isclose(obt_phys[p,q],0.0):
                coef = obt_phys[p,q]
                term = ((p,1), (q,0))
                H1 += FermionOperator(term,coef)
                if p != q:
                    H1 += hermitian_conjugated(FermionOperator(term,coef))

   #print(f'H1:')
   #print(H1)

    for p in range(N):
        for q in range(p):
            for r in range(N):
                for s in range(r+1,N):
                    term = ((p,1), (q,1), (r,0), (s,0))
                    coef_coul = tbt_phys[p,q,r,s]
                    coef_exch = tbt_phys[p,q,s,r]
                    if np.isclose(coef_coul,0.0) and np.isclose(coef_exch,0.0):
                        continue
                    else:
                        H2 += FermionOperator(term,2.0*(coef_coul - coef_exch))



    H_short = FermionOperator((),const)
    H_short += H1 + H2

   #print(f'H_short:')
   #print(H_short)

   #print(dir(H_short))
    print(f' # of terms in H_short {len(H_short.terms)}')
   #print(H_short.actions,H_short.action_strings,H_short.action_before_index)

    return(H_short)

def diff_spin(sigma_in):
    """
    Return a if sigma = b, and return b if sigma = a
    """

    if sigma_in == 'a':
        sigma_out = 'b'
    elif sigma_in == 'b':
        sigma_out = 'a'
    else:
        print('Unrecognized read in spin functions: {sigma_in}, neither a nor b')
        sys.exit()

    return sigma_out

def make_op_for_diagonal_U_space(const,obt_phys,tbt_phys,list_CIS_pair=[],debug=False):
    """
    Construct operators that couple states in the diagonal U_CSF space
    """

    if debug: print(f'\n MO pairs in CIS list: {list_CIS_pair}')

    list_SOMO = []
    for mo_pair in list_CIS_pair:
        if len(mo_pair) != 2:
            print(f'{mo_pair} does not contain two spatial mos. Bombing out!')
        if mo_pair[1] < mo_pair[0]:
            print(f'Each pair of MOs should have the smaller index as element 0 and the larger index as element 1 ')
        for mo in mo_pair:
            if mo in list_SOMO:
                print(f'duplicated mo {mo} in CIS mo pair list. Bombing out!')
                sys.exit()
            list_SOMO.append(mo)

    if debug: print(f'\n MOs in CIS list: {list_SOMO}')

    n_spinmo = obt_phys.shape[0]
    n_spatialmo = n_spinmo // 2

    tbt_unscaled = tbt_phys*2.0

    one_el_term = FermionOperator()
    diff_spin_colum_term = FermionOperator()
    same_spin_colex_term = FermionOperator()
    for p in range(n_spatialmo):
        for p_spinor in ['a','b']:
            psig = str(p)+p_spinor
            ipsig = ia_to_2i_ib_to_2iplus1(psig)
            coef = obt_phys[ipsig,ipsig]
           #if not np.isclose(coef,0.0):
            term = ((ipsig,1),(ipsig,0))
            one_el_term += FermionOperator(term,coef)

            if p_spinor == 'a':
                q_range = p+1
            else:
                q_range = p

            for q in range(q_range):
                q_spinor = diff_spin(p_spinor)
                qsig_prm = str(q) + q_spinor
                iqsig_prm = ia_to_2i_ib_to_2iplus1(qsig_prm)
                coef = tbt_unscaled[ipsig,iqsig_prm,iqsig_prm,ipsig]
               #if not np.isclose(coef,0.0):
                term = ((ipsig,1),(iqsig_prm,1),(iqsig_prm,0),(ipsig,0))
                if p == q and p in list_SOMO: continue
                diff_spin_colum_term += FermionOperator(term,coef)

                q_spinor = p_spinor
                qsig = str(q) + q_spinor
                iqsig = ia_to_2i_ib_to_2iplus1(qsig)
                coef = tbt_unscaled[ipsig,iqsig,iqsig,ipsig] - tbt_unscaled[ipsig,iqsig,ipsig,iqsig]
                term = ((ipsig,1),(iqsig,1),(iqsig,0),(ipsig,0))
               #if not np.isclose(coef,0.0):
               #if p in list_exclude and q in list_exclude: continue
                if [q,p] in list_CIS_pair: continue
                same_spin_colex_term += FermionOperator(term,coef)
    if debug:
        print(f'one_el_terms:')
        print(one_el_term)
        print(f'Coulomb terms between orbitals with different spins:')
        print(diff_spin_colum_term)
        print(f'Coulomb and Exchange terms between orbitals with same spin:')
        print(same_spin_colex_term)

    Type1_term = const+ one_el_term + diff_spin_colum_term + same_spin_colex_term
    if debug:
        print(f'Type1_term:')
        print(Type1_term)

    Type2_term = FermionOperator()
    for p in range(n_spatialmo):
        for q in range(n_spatialmo):
            if p != q:
                ipa = 2*p
                ipb = ipa+1
                iqa = 2*q
                iqb = iqa+1
                term = ((ipa,1),(ipb,1),(iqb,0),(iqa,0))
                coef = tbt_unscaled[ipa,ipb,iqb,iqa]
               #if not np.isclose(coef,0.0):
                if p in list_SOMO or q in list_SOMO: continue
                Type2_term += FermionOperator(term,coef)

    if debug:
        print(f'Type2_term:')
        print(Type2_term)

    Type3_term = FermionOperator()
   #for p in list_exclude:
   #    for q in list_exclude:
   #        if p > q:
   #            ipa = 2*p
   #            ipb = ipa+1
   #            iqa = 2*q
   #            iqb = iqa+1
   #            term = ((ipa,1),(iqb,1),(ipb,0),(iqa,0))
   #            coef = tbt_unscaled[ipa,iqb,ipb,iqa]
   #            one_term = FermionOperator(term,coef)
   #            Type3_term += one_term + hermitian_conjugated(one_term)
    for mo_pair in list_CIS_pair:
        i=mo_pair[0]
        a=mo_pair[1]
        iia = 2*i
        iib = iia+1
        iaa = 2*a
        iab = iaa+1
       #The two terms should be hermitian to each other
        term1 = ((iia,1),(iab,1),(iib,0),(iaa,0))
        coef1 = tbt_unscaled[iia,iab,iib,iaa]
        term2 = ((iib,1),(iaa,1),(iia,0),(iab,0))
        coef2 = tbt_unscaled[iib,iaa,iia,iab]
        Type3_term += FermionOperator(term1,coef1) + FermionOperator(term2,coef2)


    if debug:
        print(f'Type3_term:')
        print(Type3_term)

    nterm = len(Type1_term.terms) + len(Type2_term.terms) + len(Type3_term.terms)
    if debug:
        print(f'# of terms in make_UCSF1_terms: {nterm} ')
        print(f'# of 3 types of terms: {len(Type1_term.terms), len(Type2_term.terms), len(Type3_term.terms)}')

    return Type1_term, Type2_term, Type3_term

def make_op_for_diagonal_U_space_Stt(Enuc,obt_phys,tbt_phys,list_CIS_pair,debug=False):
    """
    Make the operator terms that contribute nonzero to the diagonal matrix elements of
    <CSF_iajb^tt|U^+ H U|CSF_iajb^tt>
    """

    if debug: print('In make_op_for_diagonal_U_space_Stt')

    lbomb = False
    if len(list_CIS_pair) != 3: lbomb = True
    if list_CIS_pair[-1] != 'tt': lbomb = True

    list_SOMO = []
    for mo_pair in list_CIS_pair[:-1]:
        if len(mo_pair) != 2:
            print(f'{mo_pair} does not contain two spatial mos. Bombing out!')
        if mo_pair[1] < mo_pair[0]:
            print(f'Each pair of MOs should have the smaller index as element 0 and the larger index as element 1 ')
        for mo in mo_pair:
            if mo in list_SOMO:
                print(f'duplicated mo {mo} in CIS mo pair list. Bombing out!')
                sys.exit()
            list_SOMO.append(mo)

    if lbomb:
        print(f'Inappropriate list_CIS_pair: {list_CIS_pair}. Bombing out')
        sys.exit()

    if debug: print(f'Singly occupied orbitals: {list_SOMO}')

    n_spinmo = obt_phys.shape[0]
    n_spatialmo = n_spinmo // 2

    tbt_unscaled = tbt_phys*2.0

    one_el_term = FermionOperator()
    diff_spin_colum_term = FermionOperator()
    same_spin_colex_term = FermionOperator()
    Type2_term = FermionOperator()

    for p in range(n_spatialmo):
        pa = 2*p
        coef = obt_phys[pa,pa]
        term = ((pa,1),(pa,0))
        one_el_term += FermionOperator(term,coef)

    one_el_term += op_spin_flip(one_el_term)

    for p in range(n_spatialmo):
        pa, pb = 2*p, 2*p+1
        if p not in list_SOMO:
            term = ((pa,1),(pb,1),(pb,0),(pa,0))
           #tbt instead of tbt_unscaled is used here because later in op_spin_flip this term will be doubled
           #coef = tbt_phys[pa,pb,pb,pa]
            coef = tbt_unscaled[pa,pb,pb,pa]
            diff_spin_colum_term += FermionOperator(term,coef)
        for q in range(p+1,n_spatialmo):
            qa, qb = 2*q, 2*q+1
            term = ((pa,1),(qa,1),(qa,0),(pa,0))
            coef = tbt_unscaled[pa,qa,qa,pa] - tbt_unscaled[pa,qa,pa,qa]
            same_spin_colex_term += FermionOperator(term,coef)
            term  = ((pa,1),(qb,1),(qb,0),(pa,0))
            term2 = ((pb,1),(qa,1),(qa,0),(pb,0))
            coef = tbt_unscaled[pa,qb,qb,pa]
            diff_spin_colum_term += FermionOperator(term,coef) + FermionOperator(term2,coef)
            if p not in list_SOMO and q not in list_SOMO:
                term = ((pa,1),(pb,1),(qb,0),(qa,0))
                coef = tbt_unscaled[pa,pb,qb,qa]
                Type2_term += FermionOperator(term,coef)

    same_spin_colex_term += op_spin_flip(same_spin_colex_term)
   #diff_spin_colum_term has been explicitly spin-adapted (term2 above)
   #diff_spin_colum_term += op_spin_flip(diff_spin_colum_term)

   #Type1_term are for each individual SD
    Type1_term = Enuc + one_el_term + same_spin_colex_term + diff_spin_colum_term

    if debug:
        print('\nType1_term:')
        print(Type1_term)

   #Type2_term are for correlation between SDs with the same occupancies in the four SOMOs
    Type2_term += hermitian_conjugated(Type2_term)

    if debug:
        print('\nType2_term:')
        print(Type2_term)

   #[[i,a],[j,b]] = list_CIS_pair[:-1]

    Type3_term = FermionOperator()
    for ip in range(len(list_SOMO)):
        p = list_SOMO[ip]
        pa, pb = 2*p, 2*p+1
        for iq in range(ip+1,len(list_SOMO)):
            q = list_SOMO[iq]
            qa, qb = 2*q, 2*q+1
            term = ((pa,1),(qb,1),(pb,0),(qa,0))
            coef = tbt_unscaled[pa,qb,pb,qa]
            Type3_term += FermionOperator(term,coef)

   #Type3_term are for the correlation between SDs with different occupancies in the four SOMOs but identical
   #pair occupations. This is the only difference from iajb_Sss
    Type3_term += op_spin_flip(Type3_term) #spin flipping also gives hermitian conjugate here

    if debug:
        print('\nType3_term:')
        print(Type3_term)

    nterm = len(Type1_term.terms) + len(Type2_term.terms) + len(Type3_term.terms)
    if debug:
        print(f'# of terms in make_op_for_diagonal_U_space_Stt: {nterm} ')
        print(f'# of 3 types of terms: {len(Type1_term.terms), len(Type2_term.terms), len(Type3_term.terms)}')


    return Type1_term, Type2_term, Type3_term


def make_op_for_offdiag_UHF_UCSFia(const,obt_phys,tbt_phys,homo,list_CIS_pair=[],debug=False):
    """
    Construct terms that contribute to the bras in UHF space and kets in UCSFia space
    """

    if debug: print('in make_op_for_offdiag_UHF_UCSFia')

    n_spinmo = obt_phys.shape[0]
    n_spatialmo = n_spinmo // 2

    tbt_unscaled = tbt_phys*2.0

    i = list_CIS_pair[0][0]
    a = list_CIS_pair[0][1]
    if i >= a:
        print(f'i >= a detected {i,a}. Bombing out!')
        sys.exit()

    list_rs = [i,a]
    list_SOMO = list_rs
    list_same_spin_rs = []
    list_diff_spin_rs = [i]
    list_same_spin_sr = []
    list_diff_spin_sr = [a]
    list_2elcorr_SOMO = []
    H_offdiag_U_1modiff = make_op_for_offdiag_U_1modiff(obt_phys,tbt_phys,list_rs,list_SOMO,list_same_spin_rs,list_diff_spin_rs,list_same_spin_sr,list_diff_spin_sr,list_2elcorr_SOMO,debug)

    return H_offdiag_U_1modiff

def make_op_for_offdiag_UHF_UCSFiajb(tbt_phys,list_CIS_pair,debug=False):
    """
    make operators that contribute to off-diagonal matrix elements between bras in U|CSF1> space
    and kets in U|CSF_iajb> space.
    list_CIS_pair = [[i,a],[j,b]]
    """

    if debug: print('\nIn make_op_for_offdiag_UHF_UCSFiajb\n')

    lbomb = False
    if len(list_CIS_pair) != 2: lbomb = True

    i, j, a, b = list_CIS_pair[0][0], list_CIS_pair[1][0], list_CIS_pair[0][1], list_CIS_pair[1][1]

    if i == j or i == a or j == b or a == b: lbomb = True
    if lbomb:
        print(f'\nInappropriate list_CIS_pair in make_op_for_offdiag_UHF_UCSFiajb: {list_CIS_pair}')
        print('Bombing out!')
        sys.exit()

    list_pqrs = [i,j,a,b]
    if debug: print(f'list_pqrs in make_op_for_offdiag_UHF_UCSFiajb_new:{list_pqrs}')

    H_offdiag_U_2modiff = make_op_for_offdiag_U_2modiff(tbt_phys,list_pqrs,debug)

    return H_offdiag_U_2modiff

def op_spin_flip(op,debug=False):
    """
    Read in an operator and flip all spin operators
    """

    if debug: print('in op_spin_flip')

    op_spinflip = FermionOperator()
    for component in op:
        if debug:
            print(f'component: {component}')
            print(f'component.terms: {component.terms}')

        for key,value in component.terms.items(): #There is only one item in component operator
            if debug: print(f'key,value:{key,value}')
            term_spinflip = ()
            for aadag  in key:
                ispin = aadag[0]
                iaction = aadag[1]
                if ispin % 2:
                    ispin_flip = ispin - 1
                else:
                    ispin_flip = ispin + 1
                aadag_spinflip = (ispin_flip,iaction)
               #print(aadag_spinflip)
               #term_spinflip.append(aadag_spinflip)
                term_spinflip += ((aadag_spinflip),)
               #print(term_spinflip)

            if debug: print(f'term_spinflip: {term_spinflip}')

        component_spinflip = FermionOperator(term_spinflip,value)
        if debug: print(f'Spin flipped component op: {component_spinflip}')
        op_spinflip += component_spinflip

    if debug:
        print(f'spin flipped operator: op_spin_flip')
        print(op_spinflip)

    return op_spinflip

def make_op_for_offdiag_UCSFia_UCSFib(obt_phys,tbt_phys,list_CIS_pair,debug=False):
    """
    make operators that contribute to off-diagonal matrix elements between bras in U|CSFia> space
    and kets in U|CSF_ib> space
    """

    if debug: print('\nIn make_op_for_offdiag_UCSFia_UCSFib')

    if len(list_CIS_pair) != 2 or list_CIS_pair[0][0] != list_CIS_pair[1][0] or list_CIS_pair[0][1] == list_CIS_pair[1][1]:
        print(f'Read in i,a and i,b pairs are not appropriate, {list_CIS_pair}, Bombing out!')
        sys.exit()

    i = list_CIS_pair[0][0]
    a = list_CIS_pair[0][1]
    b = list_CIS_pair[1][1]

    list_SOMO = [i,a,b]
    list_rs = [a,b]
    list_same_spin_rs = []
    list_diff_spin_rs = [i]
    list_same_spin_sr = [i]
    list_diff_spin_sr = [a,b]
    list_2elcorr_SOMO = [i]
    H_offdiag_U_1modiff = make_op_for_offdiag_U_1modiff(obt_phys,tbt_phys,list_rs,list_SOMO,list_same_spin_rs,list_diff_spin_rs,list_same_spin_sr,list_diff_spin_sr,list_2elcorr_SOMO,debug)

    return H_offdiag_U_1modiff

def make_op_for_offdiag_UCSFia_UCSFja(obt_phys,tbt_phys,list_CIS_pair,debug=False):
    """
    make operators that contribute to off-diagonal matrix elements between bras in U|CSFia> space
    and kets in U|CSF_ja> space
    """

    if debug: print('\nIn make_op_for_offdiag_UCSFia_UCSFja')

    if len(list_CIS_pair) != 2 or list_CIS_pair[0][1] != list_CIS_pair[1][1] or list_CIS_pair[0][0] == list_CIS_pair[1][0]:
        print(f'Inappropriate list_CIS_pair: {list_CIS_pair}. Bombing out!')
        sys.exit()

    i = list_CIS_pair[0][0]
    j = list_CIS_pair[1][0]
    a = list_CIS_pair[0][1]

    list_SOMO = [i,j,a]
    list_rs = [j,i]
    list_same_spin_rs = [a]
    list_diff_spin_rs = [i,j]
    list_same_spin_sr = []
    list_diff_spin_sr = [a]
    list_2elcorr_SOMO = [a]

    H_offdiag_U_1modiff = make_op_for_offdiag_U_1modiff(obt_phys,tbt_phys,list_rs,list_SOMO,list_same_spin_rs,list_diff_spin_rs,list_same_spin_sr,list_diff_spin_sr,list_2elcorr_SOMO,debug)

    return H_offdiag_U_1modiff

def make_op_for_offdiag_UCSFia_UCSFiajb(obt_phys,tbt_phys,list_CIS_pair,debug=False):
    """
    make operators that contribute to off-diagonal matrix elements between bras in U|CSFia> space
    and kets in U|CSF_iajb> space
    list_CIS_pair = [[i,a],[j,b]]
    """

    if debug: print('\nmake_op_for_offdiag_UCSFia_UCSFiajb')
    lbomb = False
    if len(list_CIS_pair) != 2: lbomb = True

    list_SOMO = []
    for pair in list_CIS_pair:
        for mo in pair:
            if mo in list_SOMO:
                print('Duplicated mo indices in list_CIS_pair. Bombing out!')
                lbomb = True
            else:
                list_SOMO.append(mo)

    if lbomb: sys.exit()

    i = list_CIS_pair[0][0]
    a = list_CIS_pair[0][1]
    j = list_CIS_pair[1][0]
    b = list_CIS_pair[1][1]

#Redefine list_SOMO since i and a do provide both same spin and diff spin terms in hermitian effective 1-el op
   #list_SOMO = [j,b]
    list_SOMO = [i,j,a,b]
    list_rs = [j,b]
    list_same_spin_rs = [i,a]
    list_diff_spin_rs = [i,j,a]
    list_same_spin_sr = [i,a]
    list_diff_spin_sr = [i,a,b]
    list_2elcorr_SOMO = []

    H_offdiag_U_1modiff = make_op_for_offdiag_U_1modiff(obt_phys,tbt_phys,list_rs,list_SOMO,list_same_spin_rs,list_diff_spin_rs,list_same_spin_sr,list_diff_spin_sr,list_2elcorr_SOMO,debug)


    return H_offdiag_U_1modiff

def make_op_for_offdiag_UCSFiajb_UCSFicjb(obt_phys,tbt_phys,list_CIS_pair,debug=False):
    """
    make operators that contribute to off-diagonal matrix elements between bras in U|CSFiajb> space
    and kets in U|CSF_icjb> space
    list_CIS_pair = [[i,a],[j,b],[i,c],[j,b]]
    """

    if debug: print('\nIn make_op_for_offdiag_UCSFiajb_UCSFicjb')

    lbomb = False
    if len(list_CIS_pair) != 4: lbomb = True
    if list_CIS_pair[3] != list_CIS_pair[1]: lbomb = True
    if list_CIS_pair[0][0] != list_CIS_pair[2][0] or list_CIS_pair[0][1] == list_CIS_pair[2][1]: lbomb = True
    if lbomb:
        print(f'\nInappropriate list_CIS_pair in make_op_for_offdiag_UCSFiajb_UCSFicjb: {list_CIS_pair}. Bombing out!')
        sys.exit()

    i, j    = list_CIS_pair[0][0], list_CIS_pair[1][0]
    a, b, c = list_CIS_pair[0][1], list_CIS_pair[1][1], list_CIS_pair[2][1]

    list_SOMO = [i,j,a,b,c]
    list_rs = [a,c]
    list_same_spin_rs = [j,b]
    list_diff_spin_rs = [i,j,b]
    list_same_spin_sr = [i,j,b]
    list_diff_spin_sr = [j,a,b,c]
    list_2elcorr_SOMO = [i]

    H_offdiag_U_1modiff = make_op_for_offdiag_U_1modiff(obt_phys,tbt_phys,list_rs,list_SOMO,list_same_spin_rs,list_diff_spin_rs,list_same_spin_sr,list_diff_spin_sr,list_2elcorr_SOMO,debug)

    return H_offdiag_U_1modiff

def make_op_for_offdiag_UCSFiajb_UCSFiakb(obt_phys,tbt_phys,list_CIS_pair,debug=False):
    """
    make operators that contribute to off-diagonal matrix elements between bras in U|CSFiajb> space
    and kets in U|CSF_iakb> space
    list_CIS_pair = [[i,a],[j,b],[i,a],[k,b]]
    """

    if debug: print('\nIn make_op_for_offdiag_UCSFiajb_UCSFiakb')

    lbomb = False
    if len(list_CIS_pair) != 4: lbomb = True
    if list_CIS_pair[0] != list_CIS_pair[2]: lbomb = True
    if list_CIS_pair[1][1] != list_CIS_pair[3][1] or list_CIS_pair[1][0] == list_CIS_pair[3][0]: lbomb = True
    if lbomb:
        print(f'\nInappropriate list_CIS_pair in make_op_for_offdiag_UCSFiajb_UCSFiakb: {list_CIS_pair}. Bombing out!')
        sys.exit()

    if debug: print(f'list_CIS_pair: {list_CIS_pair}')

    i, j, k = list_CIS_pair[0][0], list_CIS_pair[1][0], list_CIS_pair[3][0]
    a, b    = list_CIS_pair[0][1], list_CIS_pair[1][1]

    list_SOMO = [i,j,k,a,b]
    list_rs = [k,j]
    list_same_spin_rs = [i,a,b]
    list_diff_spin_rs = [i,j,k,a]
    list_same_spin_sr = [i,a]
    list_diff_spin_sr = [i,a,b]
    list_2elcorr_SOMO = [b]

    H_offdiag_U_1modiff = make_op_for_offdiag_U_1modiff(obt_phys,tbt_phys,list_rs,list_SOMO,list_same_spin_rs,list_diff_spin_rs,list_same_spin_sr,list_diff_spin_sr,list_2elcorr_SOMO,debug)

    return H_offdiag_U_1modiff

def make_op_for_offdiag_U_1modiff(obt_phys,tbt_phys,list_rs,list_SOMO,list_same_spin_rs,list_diff_spin_rs,list_same_spin_sr,list_diff_spin_sr,list_2elcorr_SOMO,debug=False):
    """
    Read in the info to generate operator that have nonzero contributions to the matrix elements between
    the bra and ket spaces generated by pair excitations out of reference states differ by one spatial orbital, e.g.,
    <CSF_1|U^+ H U|CSF_ia>, <CSF_ia|U^+ H U|CSF_ib>, <CSF_ia|U^+ H U|CSF_ja>, etc.
    list_rs contains the two spatial orbital indices that reflect the difference between the bra and ket reference states.
    For <CSF_1|U^+ H U|CSF_ia>, list_rs = [i,a], as <CSF_1|a_isig^+ a_asig|CSF_ia> != 0.
    For <CSF_ia|U^+ H U|CSF_ib>, list_rs = [a,b], as <CSF_ia|a_asig^+ a_bsig|CSF_ib> != 0.
    """

    if debug: print('\nIn make_op_for_offdiag_U_1modiff')

    if len(list_rs) != 2 or list_rs[0] == list_rs[1]:
        print(f'Inappropriate list_rs {list_rs}, which should contain two different spatial mo indices.')
        print('Bombing out!')
        sys.exit()

    n_spinmo = obt_phys.shape[0]
    n_spatialmo = n_spinmo // 2

    tbt_unscaled = tbt_phys*2.0
    r = list_rs[0]
    s = list_rs[1]

    ra = 2*r
    rb = ra+1
    sa = 2*s
    sb = sa+1

   #The hermitian part
    term = ((ra,1),(sa,0))
    coef = obt_phys[ra,sa]
    Term_1el = FermionOperator(term,coef)

    Term_1el += op_spin_flip(Term_1el)
    Term_1el += hermitian_conjugated(Term_1el)

    Term_eff1e_diff_spin = FermionOperator()
    Term_eff1e_same_spin = FermionOperator()
    Term_rpsp_2el_corr = FermionOperator()
    for p in range(n_spatialmo):
        if p in list_SOMO: continue
        pa = 2*p
        pb = pa+1

        term = ((ra,1),(pb,1),(pb,0),(sa,0))
        coef = tbt_unscaled[ra,pb,pb,sa]
        Term_eff1e_diff_spin += FermionOperator(term,coef)
        term = ((ra,1),(pa,1),(pa,0),(sa,0))
        coef -= tbt_unscaled[ra,pa,sa,pa]
        Term_eff1e_same_spin += FermionOperator(term,coef)
        term = ((ra,1),(sb,1),(pb,0),(pa,0))
        coef = tbt_unscaled[ra,sb,pb,pa]
        Term_rpsp_2el_corr += FermionOperator(term,coef)

    Term_2elcorr_SOMO = FermionOperator()
    for p in list_2elcorr_SOMO:
        pa = 2*p
        pb = pa+1

        term = ((ra,1),(pb,1),(sb,0),(pa,0))
        coef = tbt_unscaled[ra,pb,sb,pa]
        Term_2elcorr_SOMO += FermionOperator(term,coef)


    Term_eff1e_diff_spin += op_spin_flip(Term_eff1e_diff_spin)
    Term_eff1e_diff_spin += hermitian_conjugated(Term_eff1e_diff_spin)
    Term_eff1e_same_spin += op_spin_flip(Term_eff1e_same_spin)
    Term_eff1e_same_spin += hermitian_conjugated(Term_eff1e_same_spin)
    Term_rpsp_2el_corr += op_spin_flip(Term_rpsp_2el_corr)
    Term_rpsp_2el_corr += hermitian_conjugated(Term_rpsp_2el_corr)
    Term_2elcorr_SOMO += op_spin_flip(Term_2elcorr_SOMO)
    Term_2elcorr_SOMO += hermitian_conjugated(Term_2elcorr_SOMO)

    if debug:
        print('\nTerm_1el:')
        print(Term_1el)
        print('\nTerm_rpsp_2el_corr:')
        print(Term_rpsp_2el_corr)
        print('\nTerm_2elcorr_SOMO:')
        print(Term_2elcorr_SOMO)

   #Now the non-Hermitian part
    Term_eff1e_Ers_same_spin = FermionOperator()
    for p in list_same_spin_rs:
        pa = 2*p
        pb = pa+1
        term = ((ra,1),(pa,1),(pa,0),(sa,0))
        coef = tbt_unscaled[ra,pa,pa,sa] - tbt_unscaled[ra,pa,sa,pa]
        Term_eff1e_Ers_same_spin += FermionOperator(term,coef)

    Term_eff1e_Ers_same_spin += op_spin_flip(Term_eff1e_Ers_same_spin)

    Term_eff1e_Esr_same_spin = FermionOperator()
    for p in list_same_spin_sr:
        pa = 2*p
        pb = pa+1
        term = ((sa,1),(pa,1),(pa,0),(ra,0))
        coef = tbt_unscaled[sa,pa,pa,ra] - tbt_unscaled[sa,pa,ra,pa]
        Term_eff1e_Esr_same_spin += FermionOperator(term,coef)

    Term_eff1e_Esr_same_spin += op_spin_flip(Term_eff1e_Esr_same_spin)

    Term_eff1e_same_spin += Term_eff1e_Ers_same_spin + Term_eff1e_Esr_same_spin

    if debug:
        print('\nTerm_eff1e_same_spin:')
        print(Term_eff1e_same_spin)

    Term_eff1e_Ers_diff_spin = FermionOperator()
    for p in list_diff_spin_rs:
        pa = 2*p
        pb = pa+1
        term = ((ra,1),(pb,1),(pb,0),(sa,0))
        coef = tbt_unscaled[ra,pb,pb,sa]
        Term_eff1e_Ers_diff_spin += FermionOperator(term,coef)

    Term_eff1e_Ers_diff_spin += op_spin_flip(Term_eff1e_Ers_diff_spin)

    Term_eff1e_Esr_diff_spin = FermionOperator()
    for p in list_diff_spin_sr:
        pa = 2*p
        pb = pa+1
        term = ((sa,1),(pb,1),(pb,0),(ra,0))
        coef = tbt_unscaled[sa,pb,pb,ra]
        Term_eff1e_Esr_diff_spin += FermionOperator(term,coef)

    Term_eff1e_Esr_diff_spin += op_spin_flip(Term_eff1e_Esr_diff_spin)

    Term_eff1e_diff_spin += Term_eff1e_Ers_diff_spin + Term_eff1e_Esr_diff_spin

    if debug:
        print('\nTerm_eff1e_diff_spin:')
        print(Term_eff1e_diff_spin)

    H_offdiag_U_1modiff = Term_1el + Term_rpsp_2el_corr + Term_2elcorr_SOMO + Term_eff1e_same_spin + Term_eff1e_diff_spin

    return H_offdiag_U_1modiff

def make_op_for_offdiag_U_2modiff(tbt_phys,list_pqrs,debug=False):
    """
    Read in the info to generate operator that have nonzero contributions to the matrix elements between
    the bra and ket spaces generated by pair excitations out of reference states differ by two spatial orbital, e.g.,
    <CSF_1|U^+ H U|CSF_iajb>, <CSF_ia|U^+ H U|CSF_jb>, <CSF_iajb|U^+ H U|CSF_iakc>,  etc.
    list_pqrs = [p,q,r,s]. <CSF_bra|psig^ rsig qsig'^ ssig'|CSF_ket> != 0. sig and sig' for spin
    For <CSF_1|U^+ H U|CSF_iajb>, list_pqrs = [i,j,a,b]
    For <CSF_ia|U^+ H U|CSF_jb>, list_pqrs = [a,j,i,b]
    """

    if debug: print('\nIn make_op_for_offdiag_U_2modiff')
    p, q, r, s = list_pqrs[0],list_pqrs[1],list_pqrs[2],list_pqrs[3]
    pa, pb, qa, qb = 2*p, 2*p+1, 2*q, 2*q+1
    ra, rb, sa, sb = 2*r, 2*r+1, 2*s, 2*s+1
    if debug: print(f'p q r s in make_op_for_offdiag_U_2modiff: {p,q,r,s}')

    tbt_unscaled = tbt_phys*2.0

    H_offdiag_U_2modiff = FermionOperator()

    term = ((pa,1),(qa,1),(sa,0),(ra,0))
    coef = tbt_unscaled[pa,qa,sa,ra] - tbt_unscaled[pa,qa,ra,sa]
    H_offdiag_U_2modiff += FermionOperator(term,coef)

    term = ((pa,1),(qb,1),(sb,0),(ra,0))
    coef = tbt_unscaled[pa,qb,sb,ra]
    H_offdiag_U_2modiff += FermionOperator(term,coef)

    term = ((pa,1),(sa,1),(qa,0),(ra,0))
    coef = tbt_unscaled[pa,sa,qa,ra] - tbt_unscaled[pa,sa,ra,qa]
    H_offdiag_U_2modiff += FermionOperator(term,coef)

    term = ((pa,1),(sb,1),(qb,0),(ra,0))
    coef = tbt_unscaled[pa,sb,qb,ra]
    H_offdiag_U_2modiff += FermionOperator(term,coef)

    term = ((pa,1),(rb,1),(qb,0),(sa,0))
    coef = tbt_unscaled[pa,rb,qb,sa]
    H_offdiag_U_2modiff += FermionOperator(term,coef)

    term = ((pa,1),(rb,1),(sb,0),(qa,0))
    coef = tbt_unscaled[pa,rb,sb,qa]
    H_offdiag_U_2modiff += FermionOperator(term,coef)

    H_offdiag_U_2modiff += op_spin_flip(H_offdiag_U_2modiff)
    H_offdiag_U_2modiff += hermitian_conjugated(H_offdiag_U_2modiff)

    if debug:
        print(f'\nH_offdiag_U_2modiff:\n')
        print(H_offdiag_U_2modiff)

    return H_offdiag_U_2modiff

def make_op_for_offdiag_UCSFia_UCSFjb(tbt_phys,list_CIS_pair,debug=False):
    """
    make operators that contribute to off-diagonal matrix elements between bras in U|CSFia> space
    and kets in U|CSF_jb> space
    """

    if debug: print('\nIn make_op_for_offdiag_UCSFia_UCSFjb_new')

    lbomb = False
    if len(list_CIS_pair)!= 2: lbomb = True
    for pair in list_CIS_pair:
        if len(pair) != 2: lbomb = True

    i,a,j,b = list_CIS_pair[0][0],list_CIS_pair[0][1],list_CIS_pair[1][0],list_CIS_pair[1][1]

    if i == j or a == b or i == a or j == b: lbomb = True

    if lbomb:
        print('\nInappropriate list_CIS_pair: {list_CIS_pair}. Bombing out!')
        sys.exit()

    list_pqrs = [a,j,i,b]

    H_offdiag_U_2modiff = make_op_for_offdiag_U_2modiff(tbt_phys,list_pqrs,debug)
    return H_offdiag_U_2modiff

def make_op_for_offdiag_UCSFia_UCSFibjc(tbt_phys,list_CIS_pair,debug=False):
    """
    make operators that contribute to off-diagonal matrix elements between bras in U|CSFia> space
    and kets in U|CSF_ibjc> space.
    list_CIS_pair = [[i,a],[i,b],[j,c]]
    """

    if debug: print('\nIn make_op_for_offdiag_UCSFia_UCSFibjc')

    lbomb = False
    if len(list_CIS_pair) != 3: lbomb = True

    if list_CIS_pair[0][0] != list_CIS_pair[1][0]: lbomb = True
    i = list_CIS_pair[0][0]
    a = list_CIS_pair[0][1]
    b = list_CIS_pair[1][1]
    j = list_CIS_pair[2][0]
    c = list_CIS_pair[2][1]

    if i == j or a == b or a == c or b == c: lbomb = True

    if lbomb:
        print('\nInappropriate list_CIS_pair in make_op_for_offdiag_UCSFia_UCSFibjc. Bombing out!')
        print(list_CIS_pair)
        sys.exit()

    list_pqrs = [a,j,b,c]

    H_offdiag_U_2modiff = make_op_for_offdiag_U_2modiff(tbt_phys,list_pqrs,debug)
    return H_offdiag_U_2modiff

def make_op_for_offdiag_UCSFia_UCSFjakb(tbt_phys,list_CIS_pair,debug=False):
    """
    make operators that contribute to off-diagonal matrix elements between bras in U|CSFia> space
    and kets in U|CSF_jakb> space.
    list_CIS_pair = [[i,a],[j,a],[k,b]]
    """

    if debug: print('\nIn make_op_for_offdiag_UCSFia_UCSFjakb')

    lbomb = False
    if len(list_CIS_pair) != 3: lbomb = True

    if list_CIS_pair[0][1] != list_CIS_pair[1][1]: lbomb = True
    i = list_CIS_pair[0][0]
    a = list_CIS_pair[0][1]
    j = list_CIS_pair[1][0]
    k = list_CIS_pair[2][0]
    b = list_CIS_pair[2][1]

    if i == j or i == k or j == k or a == b: lbomb = True

    if lbomb:
        print('\nInappropriate list_CIS_pair in make_op_for_offdiag_UCSFia_UCSFjakb. Bombing out!')
        print(list_CIS_pair)
        sys.exit()

    list_pqrs = [j,k,i,b]

    H_offdiag_U_2modiff = make_op_for_offdiag_U_2modiff(tbt_phys,list_pqrs,debug)
    return H_offdiag_U_2modiff

def make_op_for_offdiag_UCSFiajb_UCSFiakc(tbt_phys,list_CIS_pair,debug=False):
    """
    make operators that contribute to off-diagonal matrix elements between bras in U|CSFiajb> space
    and kets in U|CSF_iakc> space.
    list_CIS_pair = [[i,a],[j,b],[i,a],[k,c]]
    """

    if debug: print('\nIn make_op_for_offdiag_UCSFia_UCSFjakb')

    lbomb = False
    if len(list_CIS_pair) != 4: lbomb = True

    if list_CIS_pair[0] != list_CIS_pair[2]: lbomb = True

    i,a,j,b, = list_CIS_pair[0][0],list_CIS_pair[0][1],list_CIS_pair[1][0],list_CIS_pair[1][1]
    k,c      = list_CIS_pair[3][0],list_CIS_pair[3][1]

    if i == j or i == k or j == k or i == a or j == b or k == c or a == b or a == c or b == c: lbomb = True

    if lbomb:
        print(f'\nInappropriate list_CIS_pair in make_op_for_offdiag_UCSFiajb_UCSFiakc: {list_CIS_pair}')
        print('Bombing out')
        sys.exit()

    list_pqrs = [b,k,j,c]

    H_offdiag_U_2modiff = make_op_for_offdiag_U_2modiff(tbt_phys,list_pqrs,debug)
    return H_offdiag_U_2modiff

def make_op_for_offdiag_UCSFiajb_UCSFkalb(tbt_phys,list_CIS_pair,debug=False):
    """
    make operators that contribute to off-diagonal matrix elements between bras in U|CSFiajb> space
    and kets in U|CSF_kalb> space.
    list_CIS_pair = [[i,a],[j,b],[k,a],[l,c]]
    """

    if debug: print('\nIn make_op_for_offdiag_UCSFia_UCSFkalb')

    lbomb = False
    if len(list_CIS_pair) != 4: lbomb = True

    if list_CIS_pair[0][1] != list_CIS_pair[2][1] or list_CIS_pair[1][1] != list_CIS_pair[3][1]: lbomb = True

    i,a,j,b = list_CIS_pair[0][0], list_CIS_pair[0][1], list_CIS_pair[1][0], list_CIS_pair[1][1]
    k,l     = list_CIS_pair[2][0], list_CIS_pair[3][0]

    list_SOMO = [i,j,k,l,a,b]
    for orb in list_SOMO:
        if len(np.where(list_SOMO == orb)[0]) > 1: lbomb = True

    if lbomb:
        print(f'\nInappropriate list_CIS_pair in make_op_for_offdiag_UCSFiajb_UCSFkalb: {list_CIS_pair}')
        print('Bombing out!')
        sys.exit()

    list_pqrs = [k,l,i,j]

    H_offdiag_U_2modiff = make_op_for_offdiag_U_2modiff(tbt_phys,list_pqrs,debug)
    return H_offdiag_U_2modiff

def make_op_for_offdiag_UCSFiajb_UCSFicjd(tbt_phys,list_CIS_pair,debug=False):
    """
    make operators that contribute to off-diagonal matrix elements between bras in U|CSFiajb> space
    and kets in U|CSF_icjd> space.
    list_CIS_pair = [[i,a],[j,b],[i,c],[j,d]]
    """

    if debug: print('\nIn make_op_for_offdiag_UCSFia_UCSFicjd')

    lbomb = False
    if len(list_CIS_pair) != 4: lbomb = True

    if list_CIS_pair[0][0] != list_CIS_pair[2][0] or list_CIS_pair[1][0] != list_CIS_pair[3][0]:
        lbomb = False

    i,a,j,b = list_CIS_pair[0][0], list_CIS_pair[0][1], list_CIS_pair[1][0], list_CIS_pair[1][1]
    c,d     = list_CIS_pair[2][1], list_CIS_pair[3][1]

    list_SOMO = [i,j,a,b,c,d]
    for orb in list_SOMO:
        if len(np.where(list_SOMO == orb)[0]) > 1: lbomb = True

    if lbomb:
        print(f'\nInappropriate list_CIS_pair in make_op_for_offdiag_UCSFiajb_UCSFicjd: {list_CIS_pair}')
        print('Bombing out!')
        sys.exit()

    list_pqrs = [a,b,c,d]

    H_offdiag_U_2modiff = make_op_for_offdiag_U_2modiff(tbt_phys,list_pqrs,debug)
    return H_offdiag_U_2modiff

def triplet_pair_singlet_ex(on_list,idx_list,coef_vec,list_Thp_pair,debug=False):
    """
    Given an input state with on_list, idx_list, and coef_vec, generate
    a singlet-coupled triplet pair excited state. list_Thp_pair = [[i,a],[j,b]] is a list
    of two triplet hole particle orbital pairs.
    """

    if debug: print('\nIn triplet_pair_singlet_ex')

    lbomb = False
    i,a,j,b = list_Thp_pair[0][0],list_Thp_pair[0][1],list_Thp_pair[1][0],list_Thp_pair[1][1]
    list_SOMO = [i,a,j,b]
    for orb in list_SOMO:
        if len(np.where(list_SOMO == orb)[0]) > 1: lbomb = True

    if lbomb:
        print(f'\nInappropriate list_Thp_pair in triplet_pair_singlet_ex: {list_Thp_pair}')
        print(f'Bombing out!')
        sys.exit()


    T_ia_1p1, T_ia_1_0, T_ia_1m1 = get_Tia_1m(i,a)
    T_jb_1p1, T_jb_1_0, T_jb_1m1 = get_Tia_1m(j,b)
    on_list_Tia1p1_hf, idx_list_Tia1p1_hf,coef_Tia1p1_hf = op_action_tz_remove_0coef(T_ia_1p1,on_list,idx_list,coef_vec)
    on_list_Tia1_0_hf, idx_list_Tia1_0_hf,coef_Tia1_0_hf = op_action_tz_remove_0coef(T_ia_1_0,on_list,idx_list,coef_vec)
    on_list_Tia1m1_hf, idx_list_Tia1m1_hf,coef_Tia1m1_hf = op_action_tz_remove_0coef(T_ia_1m1,on_list,idx_list,coef_vec)
    on_list_Tjb1m1Tia1p1_hf, idx_list_Tjb1m1Tia1p1_hf,coef_Tjb1m1Tia1p1_hf = \
        op_action_tz_remove_0coef(T_jb_1m1,on_list_Tia1p1_hf,idx_list_Tia1p1_hf,coef_Tia1p1_hf)
    on_list_Tjb1_0Tia1_0_hf, idx_list_Tjb1_0Tia1_0_hf,coef_Tjb1_0Tia1_0_hf = \
        op_action_tz_remove_0coef(T_jb_1_0,on_list_Tia1_0_hf,idx_list_Tia1_0_hf,coef_Tia1_0_hf)
    on_list_Tjb1p1Tia1m1_hf, idx_list_Tjb1p1Tia1m1_hf,coef_Tjb1p1Tia1m1_hf = \
        op_action_tz_remove_0coef(T_jb_1p1,on_list_Tia1m1_hf,idx_list_Tia1m1_hf,coef_Tia1m1_hf)

    coef_Tjb1m1Tia1p1_hf *= -np.sqrt(1.0/3.0)
    coef_Tjb1_0Tia1_0_hf *=  np.sqrt(1.0/3.0)
    coef_Tjb1p1Tia1m1_hf *= -np.sqrt(1.0/3.0)

    on_list_Tiajbtt_hf  = on_list_Tjb1m1Tia1p1_hf + on_list_Tjb1p1Tia1m1_hf + on_list_Tjb1_0Tia1_0_hf
    coef_Tiajbtt_hf     = np.concatenate((coef_Tjb1m1Tia1p1_hf, coef_Tjb1p1Tia1m1_hf, coef_Tjb1_0Tia1_0_hf))
    idx_list_Tiajbtt_hf = idx_list_Tjb1m1Tia1p1_hf + idx_list_Tjb1p1Tia1m1_hf + idx_list_Tjb1_0Tia1_0_hf

   #Judge whether there is any duplicate on vector?
    for idx in idx_list_Tiajbtt_hf:
        if len(np.where(idx_list_Tiajbtt_hf == idx)[0]) > 1:
            print(f'Duplicate on vector detected')
            print(on_list_Tiajbtt_hf[np.where(idx_list_Tiajbtt_hf == idx)])
            print('Bombing out!')
            sys.exit()

   #Check normalization
    coef_norm = np.linalg.norm(coef_Tiajbtt_hf)
    if not np.isclose(coef_norm,1.0):
        print(f'Non-normalized state generated by triplet_pair_singlet_ex')
        print(f'Norm = {coef_norm}')
        print(coef_Tiajbtt_hf)
        sys.exit()

    if debug:
        print(f'State generated by triplet_pair_singlet_ex:')
        for ii, item in enumerate(on_list_Tiajbtt_hf):
            print(coef_Tiajbtt_hf[ii],item)

    return on_list_Tiajbtt_hf, idx_list_Tiajbtt_hf, coef_Tiajbtt_hf

def energy_SD(Enuc,obt_phys,tbt_phys,onvec,lrdm=False,debug=False):
    """
    Calculate energy of a Slater determinant with on vec
    """

    if debug:
        print('\nIn energy_SD')
        print(f'ON: {onvec}')

   #if lrdm:
   #    n_spinorb = obt_phys.shape[0]
   #    rdm1 = np.zeros([n_spinorb,n_spinorb])
   #    rdm2 = np.zeros([n_spinorb,n_spinorb,n_spinorb,n_spinorb])
   #    rdm1 = csr_matrix((n_spinorb,n_spinorb))
   #    rdm2 = csr_matrix((n_spinorb,n_spinorb,n_spinorb,n_spinorb))
    list_rdm1 = []
    list_rdm2 = []

    spinorb_occupied = np.where(onvec != 0.0)[0]
    if debug: print(f'In energy_SD, spinorb_occupied: {spinorb_occupied}')
    E1e = 0.0
    E2e = 0.0
    for ii,iorb in enumerate(spinorb_occupied):
        E1e += obt_phys[iorb,iorb]
       #if lrdm: rdm1[iorb,iorb] += 1.0
        if lrdm: list_rdm1.append([iorb,iorb,1.0])
        for jj in range(ii+1,len(spinorb_occupied)):
            jorb = spinorb_occupied[jj]
            E2e += tbt_phys[iorb,jorb,jorb,iorb]*2.0
            if lrdm:
           #    rdm2[iorb,jorb,iorb,jorb] += 1.0
           #    rdm2[jorb,iorb,jorb,iorb] += 1.0
                list_rdm2.append([iorb,jorb,iorb,jorb,1.0])
                list_rdm2.append([jorb,iorb,jorb,iorb,1.0])
            if (iorb + jorb) % 2 == 0:
                E2e -= tbt_phys[iorb,jorb,iorb,jorb]*2.0
                if lrdm:
               #    rdm2[iorb,jorb,jorb,iorb] -= 1.0
               #    rdm2[jorb,iorb,iorb,jorb] -= 1.0
                    list_rdm2.append([iorb,jorb,jorb,iorb,-1.0])
                    list_rdm2.append([jorb,iorb,iorb,jorb,-1.0])

    E_SD = E1e + E2e + Enuc
    if debug:
        print(f'1e energy of SD: {E1e}')
        print(f'2e energy of SD: {E2e}')
        print(f'total energy of SD: {E_SD}')
       #H_short = make_short_H_ferm_op(Enuc,obt_phys,tbt_phys)
       #E_SD_check = braket_tz(onvec,onvec,H_short)
       #assert np.isclose(E_SD,E_SD_check)
       #if lrdm:
       #    E1e_from_rdm1 = elm_from_rdm1(rdm1,obt_phys)
       #    E2e_from_rdm2 = elm_from_rdm2(rdm2,tbt_phys)
       #    print(f'E1e from rdm1: {E1e_from_rdm1}')
       #    print(f'E2e from rdm2: {E2e_from_rdm2}')
       #    E_from_rdm = E1e_from_rdm1+E2e_from_rdm2 + Enuc
       #    if not np.isclose(E_SD,E_from_rdm):
       #        print(f'E_SD: {E_SD} vs. E_from_rdm: {E_from_rdm}. Bombing out!')
       #        sys.exit()
        if lrdm:
           #E1e_from_rdm1 = 0.0
           #for [i,j,rdmelm] in list_rdm1:
           #    E1e_from_rdm1 += rdmelm*obt_phys[j,i]
            E1e_from_rdm1 = elm_from_list_rdm1(list_rdm1,obt_phys)

            assert np.isclose(E1e_from_rdm1,E1e)

           #E2e_from_rdm2 = 0.0
           #for [i,j,k,l,rdmelm] in list_rdm2:
           #    E2e_from_rdm2 += rdmelm*tbt_phys[k,l,j,i]
            E2e_from_rdm2 = elm_from_list_rdm2(list_rdm2,tbt_phys)

            assert np.isclose(E2e_from_rdm2,E2e)
           #if not np.isclose(E2e_from_rdm2,E2e):
           #    print(f'E2e_from_rdm2: {E2e_from_rdm2} vs E2e: {E2e}. Bombing out!')
           #    sys.exit()


   #if lrdm:
   #    return E_SD, list_rdm1, list_rdm2
   #else:
   #    return E_SD
    return E_SD, list_rdm1, list_rdm2

def op_from_list_rdm(list_rdm1,list_rdm2,const_per_electron,obt_phys,tbt_phys):
    """
    Construct the fermionic operator that couples the bra and ket, which giveh the rdm1 and rdm2
    """

    Op1e = FermionOperator()
    Op_const = FermionOperator()
    for [q,p,rdelm] in list_rdm1:
       #print(p,q,rdelm)
        term = ((int(p),1),(int(q),0))
        coef = obt_phys[p,q]
        Op1e += FermionOperator(term,coef)
        if p == q: Op_const += FermionOperator(term,const_per_electron)

    Op2e = FermionOperator()
    for [s,r,p,q,rdelm] in list_rdm2:
        term = ((int(p),1),(int(q),1),(int(r),0),(int(s),0))
        coef = tbt_phys[p,q,r,s]
        Op2e += FermionOperator(term,coef)

    Op_const.compress()
    Op1e.compress()
    Op2e.compress()

    Op = Op_const + Op1e + Op2e
    Op.compress()
    return Op

   #term = ((pa,1),(rb,1),(qb,0),(sa,0))
   #coef = tbt_unscaled[pa,rb,qb,sa]
   #H_offdiag_U_2modiff += FermionOperator(term,coef)


def elm_from_rdm1(rdm1,obt_phys):
    """
    Calculate matrix elements given 1-el reduced density matrix and 1-el integral
    """

    return np.trace(rdm1.transpose()@obt_phys)

def elm_from_list_rdm1(list_rdm1,obt_phys):
    """
    Calculate matrix elements given a list of 1-el reduced density matrix elements and 1-el integral
    """

    E1e_from_rdm1 = 0.0
    for [i,j,rdmelm] in list_rdm1:
        E1e_from_rdm1 += rdmelm*obt_phys[j,i]

    return E1e_from_rdm1

def elm_from_rdm2(rdm2,tbt_phys):
    """
    Calculate matrix elements given 2-el reduced density matrix and 2-el integral in
    physicist notation. Formula: rdm2[i,j,k,l]*tbt_phys[k,l,j,i]
    """

    return np.einsum('ijkl,klji',rdm2,tbt_phys,optimize=True)

def elm_from_list_rdm2(list_rdm2,tbt_phys):
    """
    Calculate matrix elements given a list of 2-el reduced density matrix elements and 2-el integral
    in physicist notation. Formula: rdm2[i,j,k,l]*tbt_phys[k,l,j,i]
    """

    E2e_from_rdm2 = 0.0
    for [i,j,k,l,rdmelm] in list_rdm2:
        E2e_from_rdm2 += rdmelm*tbt_phys[k,l,j,i]

    return E2e_from_rdm2

def elm_from_list_rdm(list_rdm1,list_rdm2,const_per_electron,obt_phys,tbt_phys):
    """
    Return matrix elements from 1- and 2-el reduced density matrices
    """
    elm1e = elm_from_list_rdm1(list_rdm1,obt_phys)
    elm2e = elm_from_list_rdm2(list_rdm2,tbt_phys)
    dsum = 0.0
    for item in list_rdm1:
        if item[0] == item[1]: dsum += item[-1]

    elm_const = dsum * const_per_electron

    elm = elm_const + elm1e + elm2e
    return elm

def from_rdm_to_oper(list_rdm1,list_rdm2):
    """
    Read in rdm1 and rdm2 and return the 2nd quantized operators that give the rdms
    """

    op = FermionOperator()
    for item in list_rdm1:
        [q,p,elm] = item
        print(p,q,elm,item)
        term = ((int(p),1),(int(q),0))
        print(f'term: {term}, elm: {elm}')
        print(type(p),type(q),type(1),type(0))
        op += FermionOperator(term,elm)

    for item in list_rdm2:
        print(item)
        [s,r,p,q,elm] = item
       #if max(q,s) == max(p,r) or min(q,s) == min(p,r):
        if p in [r, s] or q in [r, s]: continue #Essentially 1-el oper multiplied by occ oper
        if p < q or s < r: continue # To remove the duplication for swapping el-1 and -2.
        term = ((int(p),1),(int(q),1),(int(r),0),(int(s),0))
        print(term,elm)
        op += FermionOperator(term,elm)

    op = normal_ordered(op)

    return op

def Helm_between_SDs_new(Enuc,obt_phys,tbt_phys,onl,onr,lrdm,debug=False):
    """
    Calculate Hamiltonian matrix element between two SDs
    """

    tic = time.perf_counter()
    if debug:
        print('\nIn Helm_between_SDs_new')
        print(f'on bra: {onl}')
        print(f'on ket: {onr}')

    on_diff = onl-onr
    on_add  = onl+onr
    assert np.isclose(np.sum(on_diff),0.0) #This guarantees conservation of number of electrons
    n_beta_dagger   = np.sum(np.where(on_diff ==  1.0)[0] % 2)
    n_beta_nodagger = np.sum(np.where(on_diff == -1.0)[0] % 2)
    if n_beta_dagger != n_beta_nodagger: return 0.0, [], []
    on_diff_norm = round(np.linalg.norm(on_diff,1))
    list_rdm1 = []
    list_rdm2 = []
    toc = time.perf_counter()
    print(f'time for initialization in Helm_between_SDs_new: {toc - tic}')
    if debug:
        print(f'on_diff: {on_diff}, on_diff_norm: {on_diff_norm}')
   #if np.isclose(on_diff_norm,0.0): #diagonal
    if on_diff_norm == 0:
        tic = time.perf_counter()
        Helm, list_rdm1, list_rdm2 = energy_SD(Enuc,obt_phys,tbt_phys,onr,lrdm,debug)
        toc = time.perf_counter()
        print(f'time for SD energy calculation in Helm_between_SDs_new: {toc - tic}')
        return Helm,list_rdm1,list_rdm2
   #elif np.isclose(on_diff_norm,2.0): #1-el
    elif on_diff_norm == 2:
        tic = time.perf_counter()
        q = np.where(on_diff == -1.0)[0][0]
        p = np.where(on_diff ==  1.0)[0][0]
       #if (p + q) % 2 == 1: #the discoincidence occurs between an alpha and a beta orbital
       #    toc = time.perf_counter()
       #    print(f'time for off-diag element with 1-pair discoin a-b in Helm_between_SDs_new: {toc - tic}')
       #    return 0.0, list_rdm1, list_rdm2
        phase = (-1.0)**np.sum(onl[min(p,q,)+1:max(p,q)])
        Helm = obt_phys[p,q]
        spin_occ_common = np.where(on_add == 2.0 )[0]
        for r in spin_occ_common:
            Helm += 2.0*(tbt_phys[p,r,r,q]-tbt_phys[p,r,q,r])
        Helm *= phase

        if lrdm:
            list_rdm1.append([q,p,phase])
            for r in spin_occ_common:
                list_rdm2.append([q,r,p,r,phase])
                list_rdm2.append([r,q,r,p,phase])
                list_rdm2.append([q,r,r,p,-phase])
                list_rdm2.append([r,q,p,r,-phase])
        toc = time.perf_counter()
        print(f'time for off-diag element with 1-pair discoin in Helm_between_SDs_new: {toc - tic}')
        return Helm, list_rdm1, list_rdm2
   #elif np.isclose(on_diff_norm,4.0): #2-el
    elif on_diff_norm == 4:
        tic = time.perf_counter()
        pq = np.where(on_diff ==  1.0)[0]
        rs = np.where(on_diff == -1.0)[0]
        p, q = pq[0], pq[1]
        r, s = rs[0], rs[1]
       #assert p < q and r < s
        phase = (-1.0)**(np.sum(onl[p+1:q]) + np.sum(onr[r+1:s]))
        if debug:
            print(f'p,q,r,s, {p,q,r,s}, phase, {phase}')
        #pqrs corresponds to p^+ q^+ s r
        #aaab, aaba, abaa, baaa, spin non-conserved
       #if (p + q + r + s) % 2 != 0 or \
       #  abs(p % 2 + q % 2 - r % 2 - s % 2) == 2: #aabb, bbaa, spin non-conserved
       #    return 0.0, list_rdm1, list_rdm2

        if p % 2 == q % 2: #aaaa or bbbb
            Helm = phase*2.0*(tbt_phys[p,q,s,r] - tbt_phys[p,q,r,s])
            if lrdm:
                list_rdm2.append([r,s,p,q,phase])
                list_rdm2.append([s,r,q,p,phase])
                list_rdm2.append([r,s,q,p,-phase])
                list_rdm2.append([s,r,p,q,-phase])
        else: #abba or baab
            #align a and b
            if p % 2 == r % 2:
                if debug: print('p, r of the same spin')
                Helm = phase*2.0*tbt_phys[p,q,s,r]
                if lrdm:
                    list_rdm2.append([r,s,p,q,phase])
                    list_rdm2.append([s,r,q,p,phase])
            else:
                if debug: print('p, r of opposite spin')
                Helm = -phase*2.0*tbt_phys[p,q,r,s]
                if lrdm:
                    list_rdm2.append([s,r,p,q,-phase])
                    list_rdm2.append([r,s,q,p,-phase])
        toc = time.perf_counter()
        print(f'time for off-diag element with 2-pair discoin in Helm_between_SDs_new: {toc - tic}')
        return Helm, list_rdm1, list_rdm2
    else:
        return 0.0, list_rdm1, list_rdm2

def Helm_between_SDs(Enuc,obt_phys,tbt_phys,onl,onr,lrdm,debug=False):
    """
    Calculate off-diagonal matrix elements between two SDs
    """

    if debug:
        print('\nIn Helm_between_SDs')
        print(f'on bra: {onl}')
        print(f'on ket: {onr}')

    Helm = 0.0
    discoin = np.where(onl != onr)[0]
    if debug: print(discoin)
    nel_l = len(np.where(onl == 1.0)[0])
    nel_r = len(np.where(onr == 1.0)[0])
    if nel_r != nel_l:
        print(f'\nInconsistent numbers of electrons in bra and ket')
        print('Bombing out')
        sys.exit()

    if len(discoin) > 4:
        if lrdm:
            return Helm, [], []
        else:
            return Helm

    if len(discoin) == 0:
       #tic = time.perf_counter()
        if lrdm:
            Helm, list_rdm1, list_rdm2 = energy_SD(Enuc,obt_phys,tbt_phys,onr,lrdm,debug)
        else:
            Helm, list_rdm1, list_rdm2 = energy_SD(Enuc,obt_phys,tbt_phys,onr,lrdm,debug)
       #toc = time.perf_counter()
       #print(f'time for energy_SD: {toc-tic}')
        if lrdm:
            return Helm, list_rdm1, list_rdm2
        else:
            return Helm

    if len(discoin) == 2:
        if lrdm:
            list_rdm1 = []
            list_rdm2 = []
        spin_occ_common = np.where((onl == onr) & (onl == 1.0) )[0]
        if debug: print(f'common occupied spin orbitals: {spin_occ_common}')
        iorb,jorb = discoin[0],discoin[1]
       #Still continue even if spin is not conserved
       #if (iorb + jorb) % 2 == 1:
       #    if lrdm:
       #        return Helm, list_rdm1,list_rdm2 #matrix elements of different spins are zero
       #    else:
       #        return Helm
        if debug: print(onl[iorb+1:jorb])
       #If there are odd commonly occupied orbitals between disjoit orbitals, flip phase
        iphase = len(np.where(onl[iorb+1:jorb] == 1.0)[0])
        phase = (-1.0)**iphase

        Helm = obt_phys[iorb,jorb]
        if debug: print(Helm)
        for porb in spin_occ_common:
            Helm += 2.0*(tbt_phys[iorb,porb,porb,jorb]-tbt_phys[iorb,porb,jorb,porb])
            if debug: print(Helm)

        Helm *= phase
        if lrdm:
       #Up to here, iorb < jorb and this does not matter for real-valued Hamiltonian.
       #But in constructing rdm, we need to clarify which of the two is an occupied
       #spin orbital in the bra (onl) and which is occupied in the ket (onr)
           #print(onl,onr)
           #print(iorb,jorb)
            if onl[iorb] > onr[iorb]:
                iiorb, jjorb = iorb, jorb
            else:
                iiorb, jjorb = jorb, iorb
           #print(iiorb,jjorb)

            list_rdm1.append([jjorb,iiorb,1.0*phase])
            for porb in spin_occ_common:
                list_rdm2.append([jjorb,porb,iiorb,porb,1.0*phase])
                list_rdm2.append([porb,jjorb,porb,iiorb,1.0*phase])
                list_rdm2.append([jjorb,porb,porb,iiorb,-1.0*phase])
                list_rdm2.append([porb,jjorb,iiorb,porb,-1.0*phase])

        if debug:
          #H_short = make_short_H_ferm_op(Enuc,obt_phys,tbt_phys)
          #Helm_check = braket_tz(onl,onr,H_short)
          #print(Helm,Helm_check)
          #assert np.isclose(Helm,Helm_check)
           if lrdm:
               Helm_1e_from_rdm1 = elm_from_list_rdm1(list_rdm1,obt_phys)
               Helm_2e_from_rdm2 = elm_from_list_rdm2(list_rdm2,tbt_phys)
               Helm_from_rdm = Helm_1e_from_rdm1 + Helm_2e_from_rdm2
               print('Testing rdm for UCSF_ia')
               assert np.isclose(Helm_from_rdm,Helm)

        if lrdm:
            return Helm, list_rdm1, list_rdm2
        else:
            return Helm

    if len(discoin) == 4:
        if lrdm:
            list_rdm1 = []
            list_rdm2 = []
        occ_l = [] #array of spin orbitals that are only occupied in bra
        occ_r = [] #array of spin orbitals that are only occupied in ket
        for ii,orb in enumerate(discoin):
            if onl[orb] == 1.0:
                if debug: print(orb)
                occ_l.append(orb)
            else:
                occ_r.append(orb)

        occ_l = np.array(occ_l)
        occ_r = np.array(occ_r)
        if debug:
            print(f'occ_l: {occ_l}, {occ_l % 2}')
            print(f'occ_r: {occ_r}, {occ_r % 2}')
        if np.sum(occ_l % 2) != np.sum(occ_r % 2):
            if debug: print(f'The 2-spin-orb disjointedness is not spin-conserved')
            Helm = 0.0

        p,q = occ_l[0],occ_l[1]
        r,s = occ_r[0],occ_r[1]
        if np.sum(occ_l % 2) == 0 or np.sum(occ_l % 2) == 2: # <aa|aa> or <bb|bb> combination
           #iphase = len(np.where(onl[p+1:r] == 1.0)[0]) + len(np.where(onl[q+1:s] == 1.0)[0])
           #phase = (-1.0)**iphase
            Helm = 2.0*(tbt_phys[p,q,s,r] - tbt_phys[p,q,r,s])
            if lrdm:
                list_rdm2.append([r,s,p,q,1.0])
                list_rdm2.append([s,r,q,p,1.0])
                list_rdm2.append([r,s,q,p,-1.0])
                list_rdm2.append([s,r,p,q,-1.0])
           #Helm *= phase
        if np.sum(occ_l % 2) == 1: #<ab|ab> combination
            if (occ_l % 2)[0] != (occ_r % 2)[0]: #swap a set of orbitals to align from <ab|ba> to <ab|ab>
                q,p = occ_l[0],occ_l[1]
           #iphase = len(np.where(onl[p+1:r] == 1.0)[0]) + len(np.where(onl[q+1:s] == 1.0)[0])
           #phase = (-1.0)**iphase
           #print(f'phase = {phase}')
            Helm = 2.0*tbt_phys[p,q,s,r]
            if lrdm:
                list_rdm2.append([r,s,p,q,1.0])
                list_rdm2.append([s,r,q,p,1.0])
           #Helm *= phase

       #
        if debug: print(f'p,q,r,s={p,q,r,s}')
        pr_min, pr_max = min(p,r), max(p,r)
        iphase = len(np.where(onl[pr_min+1:pr_max] == 1.0)[0])
        if debug: print(f'iphase = {iphase}')
        onl_tmp = copy.deepcopy(onl)
        onl_tmp[p] = 0.0
        onl_tmp[r] = 1.0
        qs_min, qs_max = min(q,s), max(q,s)
        iphase += len(np.where(onl_tmp[qs_min+1:qs_max] == 1.0)[0])
        if debug: print(f'iphase = {iphase}')
        onl_tmp[q] = 0.0
        onl_tmp[s] = 1.0
        assert np.allclose(onl_tmp,onr)
        phase = (-1.0)**iphase
        if debug: print(f'phase = {phase}')
        Helm *= phase
        if lrdm:
            for item in list_rdm2:
                item[-1] *= phase

        if debug:
       #   H_short = make_short_H_ferm_op(Enuc,obt_phys,tbt_phys)
       #   Helm_check = braket_tz(onl,onr,H_short)
       #   print(Helm,Helm_check)
       #   assert np.isclose(Helm,Helm_check)
           if lrdm:
               Helm_2e_from_rdm2 = elm_from_list_rdm2(list_rdm2,tbt_phys)
               assert np.isclose(Helm_2e_from_rdm2,Helm)

        if lrdm:
            return Helm, list_rdm1, list_rdm2
        else:
            return Helm

def Helm_between_LCSDs(Enuc,obt_phys,tbt_phys,list_onl,coefs_l,list_onr,coefs_r,lrdm=False,debug=False):
    """
    Calculate Hamiltonian matrix elements between the read-in bra and ket states (l and r)
    """
    if debug: print('\nIn Helm_between_LCSDs')

    Helm = 0.0
    if lrdm:
        list_rdm1_between_LCSDs = []
        list_rdm2_between_LCSDs = []
        # index-tuple -> position in the corresponding list, for O(1) dedup
        # (replaces the old O(n) `item[0:-1] in list_..._index` linear search).
        rdm1_pos = {}
        rdm2_pos = {}
    for ii, onl in enumerate(list_onl):
        coef_ii = coefs_l[ii]
        for jj, onr in enumerate(list_onr):
            coef_jj = coefs_r[jj]
           #Helm_between_SDs_new turns out to be slower. It is not used
           #tic = time.perf_counter()
           #Helm_new, list_rdm1_new, list_rdm2_new = Helm_between_SDs_new(Enuc,obt_phys,tbt_phys,onl,onr,lrdm,False)
           #Helm_between_SDs_new(Enuc,obt_phys,tbt_phys,onl,onr,lrdm,False)
           #toc = time.perf_counter()
           #time_new = toc - tic
            if lrdm:
               #tic = time.perf_counter()
                Helm_SDs, list_rdm1, list_rdm2 = Helm_between_SDs(Enuc,obt_phys,tbt_phys,onl,onr,lrdm,False)
               #toc = time.perf_counter()
               #time_old = toc - tic
               #print(f'time_old: {time_old}, time_new: {time_new}, time_old / time_new: {time_old / time_new}')
                if debug:
                    # Verification only: recompute the element from the RDMs and
                    # check it matches. This ran on every SD pair of every
                    # lrdm=True build; gate it behind debug so production builds
                    # skip the redundant contraction + assert.
                    elm_from_rdm1 = elm_from_list_rdm1(list_rdm1,obt_phys)
                    elm_from_rdm2 = elm_from_list_rdm2(list_rdm2,tbt_phys)
                    elm_from_rdm = elm_from_rdm1 + elm_from_rdm2
                    if np.isclose(np.sum(abs(onl - onr)),0.0): elm_from_rdm += Enuc
                    assert np.isclose(elm_from_rdm, Helm_SDs)

               #assert len(list_rdm1) == len(list_rdm1_new)
               #assert len(list_rdm2) == len(list_rdm2_new)
               #for ii in range(len(list_rdm1)):
               #    assert set(list_rdm1[ii]) == set(list_rdm1_new[ii])
               #for ii in range(len(list_rdm2)):
               #    assert set(list_rdm2[ii]) == set(list_rdm2_new[ii])
            else:
                tic = time.perf_counter()
                Helm_SDs = Helm_between_SDs(Enuc,obt_phys,tbt_phys,onl,onr,lrdm,False)
                toc = time.perf_counter()
                time_old = toc - tic
           #    print(f'time_old: {time_old}, time_new: {time_new}, time_old / time_new: {time_old / time_new}')
           #Check Helm from Helm_between_SDs_new and Helm_between_SDs
           #if not np.isclose(Helm_new,Helm_SDs):
           #    print(f'Helm_new: {Helm_new}, Helm_SDs, {Helm_SDs}')
           #    sys.exit()
            Helm += Helm_SDs*coef_ii*coef_jj
            if lrdm:
                for item in list_rdm1:
                    item[-1] *= coef_ii*coef_jj
                    key = tuple(item[0:-1])
                    pos = rdm1_pos.get(key)
                    if pos is not None:
                        list_rdm1_between_LCSDs[pos][-1] += item[-1]
                    else:
                        rdm1_pos[key] = len(list_rdm1_between_LCSDs)
                        list_rdm1_between_LCSDs.append(item)
                for item in list_rdm2:
                    item[-1] *= coef_ii*coef_jj
                    key = tuple(item[0:-1])
                    pos = rdm2_pos.get(key)
                    if pos is not None:
                        list_rdm2_between_LCSDs[pos][-1] += item[-1]
                    else:
                        rdm2_pos[key] = len(list_rdm2_between_LCSDs)
                        list_rdm2_between_LCSDs.append(item)


    if lrdm:
        return Helm, list_rdm1_between_LCSDs, list_rdm2_between_LCSDs
    else:
        return Helm

def Helm_between_CSFs(Enuc,obt_phys,tbt_phys,CSFbra,CSFket,lrdm=False):
    """
    Calculate Hamiltonian matrix elements between read in bra and ket CSFs
    """

    list_onl = CSFbra[0]
    coefs_l = CSFbra[2]
    list_onr = CSFket[0]
    coefs_r = CSFket[2]

    n_spinorb = obt_phys.shape[0]

    if lrdm:
        Helm, list_rdm1, list_rdm2 = Helm_between_LCSDs(Enuc,obt_phys,tbt_phys,list_onl,coefs_l,list_onr,coefs_r,lrdm)
        remove_zero_rdm_elements(list_rdm1)
        remove_zero_rdm_elements(list_rdm2)
        return Helm, list_rdm1, list_rdm2
    else:
        return Helm_between_LCSDs(Enuc,obt_phys,tbt_phys,list_onl,coefs_l,list_onr,coefs_r)

def make_Hmat_diagonal_space(Enuc,obt_phys,tbt_phys,list_pair_ex_space,list_pair_ex_comb,l_makeHmat=False,debug=False):
    """
    Read in a multiple pair excitation space and calculate Hamiltonian matrix elements
    """

    if debug: print('\nIn make_Hmat_diagonal_space')

    list_start_end = [[0,0]]
    ndim_space = len(list_pair_ex_space)
    ndim_pair_ex_comb = len(list_pair_ex_comb)
    list_pair_ex_comb_1col = []
    level_ex = 0
    for pair_ex_level in range(len(list_pair_ex_comb)):
        level_ex += 1
        nstate_one_ex_level = len(list_pair_ex_comb[pair_ex_level])
        start_state = list_start_end[-1][1]+1
        end_state   = list_start_end[-1][1]+nstate_one_ex_level
        list_start_end.append([start_state,end_state])
        list_pair_ex_comb_1col += list_pair_ex_comb[pair_ex_level]
        if debug: print(f'{nstate_one_ex_level} at the excitation levels of {level_ex} pair(s) of dmo and vmo')

    if debug:
        print(f'Dimension of space: {ndim_space}')
        print(f'Foldness of pair excitations: {ndim_pair_ex_comb}')
        print(f'Start and End states of each excitation foldness: {list_start_end}')
        print('One column of combination of pair excitations:')
        for ipairs, pairs in enumerate(list_pair_ex_comb_1col):
            print(ipairs+1,pairs)

    if len(list_start_end) == 1: return []
   #For each ia_pair, find the state pairs that are connected by it
    if debug:
        print(f'\nAssociating ia_pairs to state pairs\n')
        print(list_start_end[1][0],list_start_end[1][1])
    list_iapair_st_pairs = []
    list_ia_pairs = []
    for state in range(list_start_end[1][0],list_start_end[1][1]+1):
        istate = state -1
        ia_pair = list_pair_ex_comb_1col[istate][0]
        list_ia_pairs.append(ia_pair)
        if debug: print(state,istate,list_pair_ex_comb_1col[istate],ia_pair)
        list_iapair_st_pairs.append([ia_pair,[state,0]])

    if debug:
        print(f'list_iapair_st_pairs after considering pairexcitaiton level 1:')
        for item in list_iapair_st_pairs:
            print(item)
        print(f'list_ia_pairs: {list_ia_pairs}')

    for level_ex in range(2,ndim_pair_ex_comb+1):
        for state_low in range(list_start_end[level_ex-1][0],list_start_end[level_ex-1][1]+1):
            for state_up in range(list_start_end[level_ex][0],list_start_end[level_ex][1]+1):
                ii_pairs = list_pair_ex_comb_1col[state_low-1]
                jj_pairs = list_pair_ex_comb_1col[state_up-1]
                diff_mo = compare_mp2_ex_pairs(ii_pairs,jj_pairs,False)
                diff_mo = sorted(diff_mo)
                if debug:
                    print(f'Examining pairs of states {state_low,state_up} for level_ex {level_ex}, {ii_pairs,jj_pairs}')
                    print(f'diff_mo: {diff_mo}')
               #if len(diff_mo) != 2:
               #    print(f'The two states in adjacent groups do not differ by one ia pair only.')
               #else:
                if len(diff_mo) == 2:
                    if debug: print(f'States {state_low,state_up} differ by one ia pair')
                    if diff_mo in list_ia_pairs:
                        idx_ia_pair = list_ia_pairs.index(diff_mo)
                       #print(list_ia_pairs)
                       #print(f'And the different pair {diff_mo} is in the mp2 list, item {idx_ia_pair}')
                        list_iapair_st_pairs[idx_ia_pair].append([state_up,state_low])
                    diff_mo_reverse = diff_mo
                    diff_mo_reverse.reverse()
                    if diff_mo[0] == diff_mo[1]:
                        print(f'Two elements in diff_mo are identical: {diff_mo}. Bombing out!')
                        sys.exit()
                    if diff_mo_reverse in list_ia_pairs:
                        idx_ia_pair = list_ia_pairs.index(diff_mo_reverse)
                       #print(list_ia_pairs)
                       #print(f'And the different pair {diff_mo} is in the mp2 list, item {idx_ia_pair}')
                        list_iapair_st_pairs[idx_ia_pair].append([state_up,state_low])
                   #else:
                       #print(f'But the different pair is not in the mp2 list: {diff_mo}')

    if debug:
        print('\nSummary of groupping of state pairs to ia pairs')
        for item in list_iapair_st_pairs:
            print(item)

    if debug:
        for i_ia_pair,ia_pair in enumerate(list_ia_pairs):
            i,a = ia_pair[0],ia_pair[1]
            Tiiaa_00 = get_Tiiaa_00(i,a)
            list_tmp_state_pairs = [[i,a]]
            for jstate in range(ndim_space):
                state_j = list_pair_ex_space[jstate]
                onlist_j = state_j[0]
                coef_j   = state_j[2]
                for istate in range(jstate+1,ndim_space):
                    state_i = list_pair_ex_space[istate]
                    onlist_i = state_i[0]
                    coef_i   = state_i[2]
                    Tiiaa_00_elm = 0.0
                    for iSD, SDi in enumerate(onlist_i):
                        coef_SDi = coef_i[iSD]
                        for jSD, SDj in enumerate(onlist_j):
                            coef_SDj = coef_j[jSD]
                            Tiiaa_00_elm += coef_SDi*coef_SDj*braket_tz(SDi,SDj,Tiiaa_00)

                    if np.isclose(Tiiaa_00_elm,1.0): list_tmp_state_pairs.append([istate,jstate])
                    if np.isclose(Tiiaa_00_elm,-1.0):
                        print(f'Tiiaa_00_elm = -1 detected for {istate,jstate}')
                        print('This should not happen. Something wrong with the order of states.')

           #print(f'list for debugging: {list_tmp_state_pairs}')
            lpass, list_inA_notinB, list_inB_notinA = two_lists_with_same_contents(list_tmp_state_pairs,list_iapair_st_pairs[i_ia_pair])
            if not lpass:
                print(f'Different lists of couped state pairs by ia_pair: {ia_pair}')
                print(f'Stored list: {list_iapair_st_pairs[i_ia_pair]}')
                print(f'debug  list: {list_tmp_state_pairs}')
                print(list_inA_notinB)
                print(list_inB_notinA)
                print('Bombing out')
                sys.exit()

    if not l_makeHmat:
        return list_iapair_st_pairs


    H_sparse = csr_matrix((ndim_space,ndim_space))
    H_sparse_quick = csr_matrix((ndim_space,ndim_space))

    if debug:
        print('\nStates in the mp2 ex space:\n')
        for ii, item in enumerate(list_pair_ex_space):
            print(f'State {ii}: {item}')

    for ii, state in enumerate(list_pair_ex_space):
        print(f'\n ii = {ii}, {state}')
        onlist = state[0]
        coef   = state[2]
        E_state = Helm_between_LCSDs(Enuc,obt_phys,tbt_phys,onlist,coef,onlist,coef,False,False)
        H_sparse[ii,ii] = E_state
        H_sparse_quick[ii,ii] = E_state
        if ii == 0: continue
       #Check ii belongs to which group
        for igroup,group in enumerate(list_start_end):
            if ii >= group[0] and ii <= group[1]:
               #print(f'{ii} belongs to group {igroup}')
               break
        jgroup = igroup - 1
       #The following chunk still involves Helm_between_LCSDs and can be slow
        jj_start = list_start_end[jgroup][0]
        jj_end   = ii
        for jj in range(jj_start,jj_end):
            state_jj = list_pair_ex_space[jj]
            onlist_jj = state_jj[0]
            coef_jj   = state_jj[2]
           #print(f'Calculating States {jj,ii}, jj from {jj_start} to {jj_end}')
           #print(f'onlist_jj: {onlist_jj}')
           #print(f'coef_jj: {coef_jj}')
           #print(f'onlist: {onlist}')
           #print(f'coef: {coef}')
            Helm = Helm_between_LCSDs(Enuc,obt_phys,tbt_phys,onlist,coef,onlist_jj,coef_jj,False,False)
           #print(f'Calculated H{jj,ii}: {Helm}')
            if not np.isclose(Helm,0.0):
                H_sparse[jj,ii] = Helm
                H_sparse[ii,jj] = Helm

       #The following chunk only examines discoincidence of excitation pairs
        jj_start = list_start_end[jgroup][0]
       #jj_end   = list_start_end[jgroup][1]+1
        jj_end   = ii
        ii_pos   = ii - 1
        ii_pairs = list_pair_ex_comb_1col[ii_pos]
        if igroup == 1:
            dmo = ii_pairs[0][0]
            vmo = ii_pairs[0][1]
            dmoa,dmob,vmoa,vmob = dmo*2,dmo*2+1,vmo*2,vmo*2+1
            Helm_quick = 2.0*tbt_phys[dmoa,dmob,vmob,vmoa]
            if not np.isclose(Helm_quick,0.0):
                H_sparse_quick[ii,0] = Helm_quick
                H_sparse_quick[0,ii] = Helm_quick
            print(f'Helm_quick for {jj,ii}: {Helm_quick}')
            jj_start = list_start_end[igroup][0]
        for jj in range(jj_start,jj_end):
            jj_pos = jj-1
            jj_pairs = list_pair_ex_comb_1col[jj_pos]
            diff_mo = compare_mp2_ex_pairs(ii_pairs,jj_pairs,False)
               #for ipair, ii_pair in enumerate(ii_pairs):
               #    if ii_pair not in jj_pairs:
               #        n_ii_pair_not_in_jj_pairs += 1
               #        pair_new = ii_pair
               #if n_ii_pair_not_in_jj_pairs > 1:
               #    print(f'{jj,ii}, {jj_pairs}, {ii_pairs}, differ by {n_ii_pair_not_in_jj_pairs} pairs. Skipping!')
               #    continue
               #print(f'{jj,ii}, {jj_pairs}, {ii_pairs}, differ by one pair: {pair_new}')
            if len(diff_mo) != 2:
                if debug: print(f'Skipping states {jj,ii} for > 1 pair of different MOs: {diff_mo}')
                continue
            dmo = diff_mo[0]
            vmo = diff_mo[1]
            dmoa,dmob,vmoa,vmob = dmo*2,dmo*2+1,vmo*2,vmo*2+1
            Helm_quick = 2.0*tbt_phys[dmoa,dmob,vmob,vmoa]
            if not np.isclose(Helm_quick,0.0):
                H_sparse_quick[ii,jj] = Helm_quick
                H_sparse_quick[jj,ii] = Helm_quick
                print(f'Helm_quick for {jj,ii}: {Helm_quick}')


    print('\nH_sparse:\n')
    print(H_sparse)

    if debug:
        H_sparse_check = csr_matrix((ndim_space,ndim_space))
        for ii in range(ndim_space):
            state_ii = list_pair_ex_space[ii]
            onlist_ii = state_ii[0]
            coef_ii   = state_ii[2]
            E_state = Helm_between_LCSDs(Enuc,obt_phys,tbt_phys,onlist_ii,coef_ii,onlist_ii,coef_ii,False,False)
            H_sparse_check[ii,ii] = E_state
            for jj in range(ii):
                state_jj = list_pair_ex_space[jj]
                onlist_jj = state_jj[0]
                coef_jj   = state_jj[2]
                Helm = Helm_between_LCSDs(Enuc,obt_phys,tbt_phys,onlist_ii,coef_ii,onlist_jj,coef_jj,False,False)
                if not np.isclose(Helm,0.0):
                    H_sparse_check[ii,jj] = Helm
                    H_sparse_check[jj,ii] = Helm

       #print('\nH_sparse_check:\n')
       #print(H_sparse_check)
        norm_diff = scipy.sparse.linalg.norm(H_sparse - H_sparse_check)
        assert np.isclose(norm_diff,0.0)
        print('\nChecking H_sparse vs H_sparse_quick')
       #print('\nH_sparse_quick:\n')
        print(H_sparse_quick)
        norm_diff = scipy.sparse.linalg.norm(H_sparse - H_sparse_quick)
       #print('\nH_sparse - H_sparse_quick:\n')
       #print(H_sparse - H_sparse_quick)
        assert np.isclose(norm_diff,0.0)

    return H_sparse_quick, list_iapair_st_pairs

def obt_phys_spatial_to_spin(obt_phys_spatial):
    """
    Convert the obt of spatial orbitals to obt of spin orbitals
    """

    n_spatial = obt_phys_spatial.shape[0]
    n_spin = 2*n_spatial
    obt_phys_spin = np.zeros([n_spin,n_spin])
    obt_phys_spin[0:n_spin:2,0:n_spin:2] = obt_phys_spatial
    obt_phys_spin[1:n_spin:2,1:n_spin:2] = obt_phys_spatial

    return obt_phys_spin

def obt_phys_spin_to_spatial(obt_phys_spin):
    """
    Convert the obt of spin orbitals to obt of spatial orbitals
    """

    n_spin = obt_phys_spin.shape[0]
    n_spatial = n_spin // 2
    obt_phys_spatial = np.zeros([n_spatial,n_spatial])
    obt_phys_spatial = obt_phys_spin[0:n_spin:2,0:n_spin:2]

    return obt_phys_spatial

def tbt_phys_spatial_to_spin(tbt_phys_spatial):
    """
    Convert the tbt in physicist notation of spatial orbitals
    to tbt of spin orbitals in physicist notation
    """

    n_spatial = tbt_phys_spatial.shape[0]
    n_spin = 2*n_spatial
    tbt_phys_spin = np.zeros([n_spin,n_spin,n_spin,n_spin])
    tbt_phys_spin[0:n_spin:2,0:n_spin:2,0:n_spin:2,0:n_spin:2] = tbt_phys_spatial
    tbt_phys_spin[1:n_spin:2,1:n_spin:2,1:n_spin:2,1:n_spin:2] = tbt_phys_spatial
    tbt_phys_spin[0:n_spin:2,1:n_spin:2,1:n_spin:2,0:n_spin:2] = tbt_phys_spatial
    tbt_phys_spin[1:n_spin:2,0:n_spin:2,0:n_spin:2,1:n_spin:2] = tbt_phys_spatial

    return tbt_phys_spin

def tbt_phys_spin_to_spatial(tbt_phys_spin):
    """
    Convert the tbt in physicist notation of spin orbitals
    to tbt of spatial orbitals in physicist notation
    """

    n_spin = tbt_phys_spin.shape[0]
    n_spatial = n_spin // 2

    tbt_phys_spatial = np.zeros([n_spatial,n_spatial,n_spatial,n_spatial])
    tbt_phys_spatial = tbt_phys_spin[0:n_spin:2,0:n_spin:2,0:n_spin:2,0:n_spin:2]

    return tbt_phys_spatial

def compare_mp2_ex_pairs(ii_pairs,jj_pairs,debug=False):
    """
    Read in two lists of lists like [[dmo1,vmo1],[dmo2,vmo2],...]
    Compare them and return the discoincidence of dmo and vmo
    """

    if debug:
        print('\nIn compare_mp2_ex_pairs')
        print(f'ii_pairs: {ii_pairs}, jj_pairs: {jj_pairs}')
    npair_ii = len(ii_pairs)
    npair_jj = len(jj_pairs)
    ii_exmo = []
    for pair in ii_pairs:
        ii_exmo += pair
   #ii_exmo = sorted(ii_exmo)
   #norb = len(ii_exmo)
   #ii_exdmo = ii_exmo[:norb//2]
   #ii_exvmo = ii_exmo[norb//2:]
    ii_exdmo = []
    ii_exvmo = []
    for pair in ii_pairs:
        ii_exdmo.append(pair[0])
        ii_exvmo.append(pair[1])
    jj_exmo = []
    for pair in jj_pairs:
        jj_exmo += pair

   #jj_exmo = sorted(jj_exmo)
   #norb = len(jj_exmo)
   #jj_exdmo = jj_exmo[:norb//2]
   #jj_exvmo = jj_exmo[norb//2:]
    jj_exdmo = []
    jj_exvmo = []
    for pair in jj_pairs:
        jj_exdmo.append(pair[0])
        jj_exvmo.append(pair[1])
    if debug:
        print(f'ii_exmo: {ii_exmo}, ii_exdmo: {ii_exdmo}, ii_exvmo: {ii_exvmo}')
        print(f'jj_exmo: {jj_exmo}, jj_exdmo: {jj_exdmo}, jj_exvmo: {jj_exvmo}')
    dmo_in_ii_not_in_jj = list(filter(lambda x: x not in jj_exdmo, ii_exdmo))
    dmo_in_jj_not_in_ii = list(filter(lambda x: x not in ii_exdmo, jj_exdmo))
    vmo_in_ii_not_in_jj = list(filter(lambda x: x not in jj_exvmo, ii_exvmo))
    vmo_in_jj_not_in_ii = list(filter(lambda x: x not in ii_exvmo, jj_exvmo))
    if debug: print(dmo_in_ii_not_in_jj,dmo_in_jj_not_in_ii,vmo_in_ii_not_in_jj,vmo_in_jj_not_in_ii)
   #if npair_ii == npair_jj:
   #    if len(dmo_in_ii_not_in_jj) != len(dmo_in_jj_not_in_ii) or \
   #       len(vmo_in_ii_not_in_jj) != len(vmo_in_jj_not_in_ii):
   #       print(f'Strange! Dimensions not matched')
   #       print(dmo_in_ii_not_in_jj,dmo_in_jj_not_in_ii,vmo_in_ii_not_in_jj,vmo_in_jj_not_in_ii)
   #       print('Bombing out')
   #       sys.exit()
    if abs(len(dmo_in_ii_not_in_jj) - len(dmo_in_jj_not_in_ii)) != abs(npair_ii - npair_jj) or\
       abs(len(vmo_in_ii_not_in_jj) - len(vmo_in_jj_not_in_ii)) != abs(npair_ii - npair_jj):
           print(f'Strange! Dimensions not matched')
           print(dmo_in_ii_not_in_jj,dmo_in_jj_not_in_ii,vmo_in_ii_not_in_jj,vmo_in_jj_not_in_ii)
           print('Bombing out')
           sys.exit()

    diff_pair = vmo_in_ii_not_in_jj + vmo_in_jj_not_in_ii + dmo_in_ii_not_in_jj + dmo_in_jj_not_in_ii
    if debug:
        print(f'diff_pair: {diff_pair}')
   #if len(diff_pair) != 2 and debug:
   #    print(f'Difference more than two doubly occupied orbitals {diff_pair}. Matrix element = 0')

    return diff_pair

def two_lists_with_same_contents(listA,listB,debug=False):
    """
    Judge whether two lists have the same contents, regardless of their orders
    """

    lsame = False
    elm_in_A_not_in_B = list(filter(lambda x: x not in listB, listA))
    elm_in_B_not_in_A = list(filter(lambda x: x not in listA, listB))
    if len(listA) != len(listB):
        if debug:
            print('The two lists do not even have the same length')
            print(f'listA: {listA}')
            print(f'listB: {listB}')
        return lsame, elm_in_A_not_in_B, elm_in_B_not_in_A

    if len(elm_in_A_not_in_B) != 0 or len(elm_in_B_not_in_A) != 0:
        if debug:
            print('The two lists do not contain the same elements')
            print(f'Those in A but not in B: {elm_in_A_not_in_B}')
            print(f'Those in B but not in A: {elm_in_B_not_in_A}')
        return lsame,elm_in_A_not_in_B,elm_in_B_not_in_A

    lsame = True
    return lsame, [], []

def overlap_LCSD(onlist_l,idx_list_l,coefs_l,onlist_r,idx_list_r,coefs_r,debug=False):
    """
    Calculate overlap between two states as linear combinations of SDs
    Quick calculation is done using the indices lists. The on vector lists are used for
    debugging only
    """

    if debug: print('\nIn overlap_LCSD')

    ovrlap = 0.0
    for ii, idx_l in enumerate(idx_list_l):
        coefl = coefs_l[ii]
        for jj, idx_r in enumerate(idx_list_r):
            coefr = coefs_r[jj]
            if idx_l == idx_r: ovrlap += coefl*coefr

    return ovrlap

def overlap_CSFs(CSFbra,CSFket,debug=False):
    """
    Calculate overlap between two CSFs
    """

    [onlist_l, idx_list_l, coefs_l] = CSFbra
    [onlist_r, idx_list_r, coefs_r] = CSFket

    return overlap_LCSD(onlist_l,idx_list_l,coefs_l,onlist_r,idx_list_r,coefs_r,debug)


def make_UCSF_state(list_ex_states,coefs_ex_states,debug=False):

    if debug: print('\nIn make_UCSF_state')

    idx_list_output = []
    onlist_output = []
    coef_output = []
    for ii, state in enumerate(list_ex_states):
        coef_state = coefs_ex_states[ii]
        if abs(coef_state) < 1e-15:
            continue   # skip zero-weight states so idx_list stays compact and unique
        onlist = state[0]
        idx_list = state[1]
        coef_vec = state[2]
        onlist_output += onlist
        idx_list_output += idx_list
        coef_output += (coef_vec*coef_state).tolist()

    coef_output = np.array(coef_output)
    if debug:
        for ii in range(len(coef_output)):
            print(onlist_output[ii],idx_list_output[ii],coef_output[ii])
       #Check normalization
        norm_state = np.linalg.norm(coef_output)
        assert np.isclose(norm_state,1.0)
        norm_state_2nd = overlap_LCSD(onlist_output,idx_list_output,coef_output,onlist_output,idx_list_output,coef_output)
        assert np.isclose(norm_state_2nd,1.0)

    return [onlist_output,idx_list_output,coef_output]

def prepare_UCSFia(list_bound,ref_onlist,ref_idxlist,ref_coefvec,mp2_ia_pairs,group_mp2_ia_pairs,group_mp2_ampld,debug=False):
    """
    Prepare U|CSFia> states, given the read-in information of a HF reference
    list_bound = [i_low,i_up,a_low,a_up], the lower and upper bounds of hole and particle orbitals
    """

    if debug: print('\nIn prepare_UCSFia')

    UCSF_ia_basis = []
    UCSF_ia_CIS_list = []
    UCSF_ia_vec_list = []
    UCSF_ia_list_ex_space = []
    UCSF_ia_list_list_genmat = []
    [i_low, i_up, a_low, a_up] = list_bound
    for i in range(i_up,i_low-1,-1):
        for a in range(a_low,a_up+1):
            T_ia_00 = get_Tia_00(i,a)
            on_list_Tia_hf, idx_list_Tia_hf,coef_Tia_hf = op_action_tz_remove_0coef(T_ia_00,ref_onlist,ref_idxlist,ref_coefvec)
            list_ia_pair_ex_space, list_ia_pair_ex_comb = make_pair_ex_space_fast(on_list_Tia_hf,idx_list_Tia_hf,coef_Tia_hf,mp2_ia_pairs)
            list_iapair_st_pairs = make_Hmat_diagonal_space(None,None,None,list_ia_pair_ex_space,list_ia_pair_ex_comb,False,False)
            list_genmat = make_iapair_genmat_fast(group_mp2_ia_pairs,list_ia_pair_ex_space,list_iapair_st_pairs,False)
            if len(list_genmat) == 0:
                ndim_space = len(list_ia_pair_ex_space)
                Umat_ia = np.eye(ndim_space)
                U_ia_vec = np.zeros([ndim_space])
                U_ia_vec[0] = 1.0
            else:
                Umat_ia, U_ia_vec = make_and_apply_U_matrix(list_genmat,group_mp2_ampld)
            Uia_state = make_UCSF_state(list_ia_pair_ex_space,U_ia_vec)
            UCSF_ia_basis.append(Uia_state)
            UCSF_ia_CIS_list.append([[i,a]])
            UCSF_ia_vec_list.append(U_ia_vec)
            UCSF_ia_list_ex_space.append(list_ia_pair_ex_space)
            UCSF_ia_list_list_genmat.append(list_genmat)

    return UCSF_ia_basis, UCSF_ia_CIS_list, UCSF_ia_vec_list, UCSF_ia_list_ex_space, UCSF_ia_list_list_genmat

def prepare_UCSFiajb_Sss(list_bound,ref_onlist,ref_idxlist,ref_coefvec,mp2_ia_pairs,group_mp2_ia_pairs,group_mp2_ampld,debug=False):
    """
    Prepare U|CSFiajb_Sss> states, given the read-in information of a HF reference
    list_bound = [i_low,i_up,a_low,a_up], the lower and upper bounds of hole and particle orbitals
    """

    if debug: print('\nIn prepare_UCSFia')

    UCSF_iajb_basis = []
    UCSF_iajb_CIS_list = []
    UCSF_iajb_vec_list = []
    UCSF_iajb_list_pair_ex = []
    UCSF_iajb_list_Umat = []
    UCSF_iajb_list_list_genmat = []
    [i_low, i_up, a_low, a_up] = list_bound
    for i in range(i_up,i_low-1,-1):
        for a in range(a_low,a_up+1):
            T_ia_00 = get_Tia_00(i,a)
            on_list_Tia_hf, idx_list_Tia_hf,coef_Tia_hf = op_action_tz_remove_0coef(T_ia_00,ref_onlist,ref_idxlist,ref_coefvec)
            for j in range(i-1,i_low-1,-1):
                for b in range(a+1,a_up+1):
                    T_jb_00 = get_Tia_00(j,b)
                    on_list_Tiajb_hf, idx_list_Tiajb_hf,coef_Tiajb_hf = op_action_tz_remove_0coef(T_jb_00,on_list_Tia_hf,idx_list_Tia_hf,coef_Tia_hf)
                    list_iajb_pair_ex_space, list_iajb_pair_ex_comb = make_pair_ex_space_fast(on_list_Tiajb_hf,idx_list_Tiajb_hf,coef_Tiajb_hf,mp2_ia_pairs)
                    list_iapair_st_pairs = make_Hmat_diagonal_space(None,None,None,list_iajb_pair_ex_space,list_iajb_pair_ex_comb,False,False)
                    list_genmat = make_iapair_genmat_fast(group_mp2_ia_pairs,list_iajb_pair_ex_space,list_iapair_st_pairs,False)
                    if len(list_genmat) == 0:
                        ndim_space = len(list_iajb_pair_ex_space)
                        Umat_iajb = np.eye(ndim_space)
                        U_iajb_vec = np.zeros([ndim_space])
                        U_iajb_vec[0] = 1.0
                    else:
                        Umat_iajb, U_iajb_vec = make_and_apply_U_matrix(list_genmat,group_mp2_ampld)
                    Uiajb_state = make_UCSF_state(list_iajb_pair_ex_space,U_iajb_vec)
                    UCSF_iajb_basis.append(Uiajb_state)
                    UCSF_iajb_CIS_list.append([[i,a],[j,b]])
                    UCSF_iajb_vec_list.append(U_iajb_vec)
                    UCSF_iajb_list_pair_ex.append(list_iajb_pair_ex_space)
                    UCSF_iajb_list_Umat.append(Umat_iajb)
                    UCSF_iajb_list_list_genmat.append(list_genmat)

    return UCSF_iajb_basis, UCSF_iajb_CIS_list, UCSF_iajb_vec_list, UCSF_iajb_list_pair_ex, UCSF_iajb_list_Umat, UCSF_iajb_list_list_genmat

def prepare_UCSFiajb_Stt(list_bound,ref_onlist,ref_idxlist,ref_coefvec,mp2_ia_pairs,group_mp2_ia_pairs,group_mp2_ampld,debug=False):
    """
    Prepare U|CSFiajb_Stt> states, given the read-in information of a HF reference
    list_bound = [i_low,i_up,a_low,a_up], the lower and upper bounds of hole and particle orbitals
    """

    if debug: print('\nIn prepare_UCSFia')

    UCSF_iajb_basis = []
    UCSF_iajb_CIS_list = []
    UCSF_iajb_vec_list = []
    UCSF_iajb_list_pair_ex = []
    Umat_iajb_list = []
    UCSF_iajb_list_list_genmat = []
    [i_low, i_up, a_low, a_up] = list_bound
    for i in range(i_up,i_low-1,-1):
        for a in range(a_low,a_up+1):
            for j in range(i-1,i_low-1,-1):
                for b in range(a+1,a_up+1):
                    list_Thp_pair = [[i,a],[j,b]]
                    on_list_Tiajb_hf, idx_list_Tiajb_hf,coef_Tiajb_hf = triplet_pair_singlet_ex(ref_onlist,ref_idxlist,ref_coefvec,list_Thp_pair)
                    list_iajb_pair_ex_space, list_iajb_pair_ex_comb = make_pair_ex_space_fast(on_list_Tiajb_hf,idx_list_Tiajb_hf,coef_Tiajb_hf,mp2_ia_pairs)
                    list_iapair_st_pairs = make_Hmat_diagonal_space(None,None,None,list_iajb_pair_ex_space,list_iajb_pair_ex_comb,False,False)
                    list_genmat = make_iapair_genmat_fast(group_mp2_ia_pairs,list_iajb_pair_ex_space,list_iapair_st_pairs,False)
                    if len(list_genmat) == 0:
                        ndim_space = len(list_iajb_pair_ex_space)
                        Umat_iajb = np.eye(ndim_space)
                        U_iajb_vec = np.zeros([ndim_space])
                        U_iajb_vec[0] = 1.0
                    else:
                        Umat_iajb, U_iajb_vec = make_and_apply_U_matrix(list_genmat,group_mp2_ampld)
                    Uiajb_state = make_UCSF_state(list_iajb_pair_ex_space,U_iajb_vec)
                    UCSF_iajb_basis.append(Uiajb_state)
                    UCSF_iajb_CIS_list.append([[i,a],[j,b],'tt'])
                    UCSF_iajb_vec_list.append(U_iajb_vec)
                    UCSF_iajb_list_pair_ex.append(list_iajb_pair_ex_space)
                    Umat_iajb_list.append(Umat_iajb)
                    UCSF_iajb_list_list_genmat.append(list_genmat)

    return UCSF_iajb_basis, UCSF_iajb_CIS_list, UCSF_iajb_vec_list, UCSF_iajb_list_pair_ex, Umat_iajb_list, UCSF_iajb_list_list_genmat

def prepare_UCSF_generic(list_CSF,sorted_mp2_ia_pairs,group_mp2_ia_pairs,group_mp2_ampld,nparal=1,debug=False):
    """
    Prepare U|CSF> for a read-in set of CSFs.
    """

    if debug: print('\nIn prepare_UCSF_generic')

    UCSF_basis = []
   #UCSF_SOMO_list = []
    Uvec_list = []
    UCSF_list_list_pair_ex_space = []
    Umat_list = []
    UCSF_list_list_genmat = []

    for ii, CSF in enumerate(list_CSF):
       #if ii % 10 == 0: print(f'Making UCSF for State {ii} of {len(list_CSF)} states')
        [onlist, onidx_list, coefvec] = CSF
       #list_pair_ex_space, list_pair_ex_comb = make_pair_ex_space_fast(onlist,onidx_list,coefvec,sorted_mp2_ia_pairs)
       #list_iapair_st_pairs = make_Hmat_diagonal_space(None,None,None,list_pair_ex_space,list_pair_ex_comb,False,True)
       #print(f'list_iapair_st_pairs:')
       #for item in list_iapair_st_pairs:
       #    print(item)
       #list_genmat = make_iapair_genmat_fast(group_mp2_ia_pairs,list_pair_ex_space,list_iapair_st_pairs,True)
       #list_pair_ex_space, list_genmat = make_pair_ex_space_manual(CSF,sorted_mp2_ia_pairs,debug=False)
        list_pair_ex_space, list_genmat = make_pair_ex_space_manual(CSF,group_mp2_ia_pairs,nparal=nparal,debug=False)
       #print(f'list_genmat:')
       #for genmat in list_genmat:
       #    print(genmat)
        if len(list_genmat) == 0:
            ndim_space = len(list_pair_ex_space)
            Umat = np.eye(ndim_space)
            Uvec = np.zeros([ndim_space])
            Uvec[0] = 1.0
        else:
            Umat, Uvec = make_and_apply_U_matrix(list_genmat,group_mp2_ampld)

        Ustate = make_UCSF_state(list_pair_ex_space,Uvec)
        UCSF_basis.append(Ustate)
        Uvec_list.append(Uvec)
        UCSF_list_list_pair_ex_space.append(list_pair_ex_space)
        Umat_list.append(Umat)
        UCSF_list_list_genmat.append(list_genmat)

    return UCSF_basis, Uvec_list, UCSF_list_list_pair_ex_space, Umat_list, UCSF_list_list_genmat

def prepare_UCSF_group_generic(list_CSF,sorted_mp2_ia_pairs, group_mp2_ia_pairs,group_mp2_ampld, extra_basis_CSF = [], nparal=1,debug=False):
    """
    Prepare U|CSF> for a read-in set of CSFs.
    """

    if debug: print('\nIn prepare_UCSF_generic')

    UCSF_basis = []
   #UCSF_SOMO_list = []
    Uvec_list = []
    UCSF_list_list_pair_ex_space = []
    Umat_list = []
    UCSF_list_list_genmat = []

    for ii, CSF in enumerate(list_CSF):
       #if ii % 10 == 0: print(f'Making UCSF for State {ii} of {len(list_CSF)} states')
        [onlist, onidx_list, coefvec] = CSF
       #list_pair_ex_space, list_pair_ex_comb = make_pair_ex_space_fast(onlist,onidx_list,coefvec,sorted_mp2_ia_pairs)
       #list_iapair_st_pairs = make_Hmat_diagonal_space(None,None,None,list_pair_ex_space,list_pair_ex_comb,False,True)
       #print(f'list_iapair_st_pairs:')
       #for item in list_iapair_st_pairs:
       #    print(item)
       #list_genmat = make_iapair_genmat_fast(group_mp2_ia_pairs,list_pair_ex_space,list_iapair_st_pairs,True)
       #list_pair_ex_space, list_genmat = make_pair_ex_space_manual(CSF,sorted_mp2_ia_pairs,debug=False)
       #list_pair_ex_space, list_genmat = make_pair_ex_space_manual(CSF,group_mp2_ia_pairs,nparal=nparal,debug=False)
        current_extra = extra_basis_CSF[ii] if extra_basis_CSF else []

        # Update this call to pass the extra basis to the manual builder
        list_pair_ex_space, list_genmat = make_pair_ex_space_group_manual(
            CSF,
            group_mp2_ia_pairs,
            extra_basis_CSF=current_extra,
            nparal=nparal,
            debug=False
        )
       #print(f'list_genmat:')
       #for genmat in list_genmat:
       #    print(genmat)
        if len(list_genmat) == 0:
            ndim_space = len(list_pair_ex_space)
            Umat = np.eye(ndim_space)
            Uvec = np.zeros([ndim_space])
            Uvec[0] = 1.0
        else:
            Umat, Uvec = make_and_apply_U_matrix(list_genmat,group_mp2_ampld)

        Ustate = make_UCSF_state(list_pair_ex_space,Uvec)
        UCSF_basis.append(Ustate)
        Uvec_list.append(Uvec)
        UCSF_list_list_pair_ex_space.append(list_pair_ex_space)
        Umat_list.append(Umat)
        UCSF_list_list_genmat.append(list_genmat)

    return UCSF_basis, Uvec_list, UCSF_list_list_pair_ex_space, Umat_list, UCSF_list_list_genmat

def select_ia_pairs_for_CSF_E0(list_CSF,list_ex_space,group_ia_pairs,Enuc,obt,tbt,Ethrsh=1.0e-3,list_list_ia_included=[],debug=False):
    """
    Select ia pairs for each CSF for their individual lowerings of the E0 of the CSF space
    """

    if debug: print('\nIn select_ia_pairs_for_CSF_E0')

    Hmat_ex = construct_Hmat_CSFs(list_ex_space,Enuc,obt,tbt)

    Hmat_ex_sparse = csr_matrix(Hmat_ex)
    E0_ex, _ = get_ground_state(Hmat_ex_sparse)
    print(f'\nE0 of original ex space: {E0_ex}')
    nCSF = len(list_CSF)
    if len(list_list_ia_included) == 0:
        for iCSF in range(nCSF):
            list_list_ia_included.append([])
    list_list_genmat = []
    list_list_ex_space = []
    for iCSF, CSF in enumerate(list_CSF):
        print(f'Working on CSF{iCSF}')
        list_ia_one_CSF = []
        list_improve = []
        for ia_pair in group_ia_pairs:
            if ia_pair in list_list_ia_included[iCSF]: continue
            UCSF, Uvec, list_pair_ex_space, Umat, _ = \
              prepare_UCSF_for_one_CSF(CSF,[ia_pair],[0.0],False)
            list_UCSF_space = list_ex_space + list_pair_ex_space[1:]
           #Hmat_UCSF = construct_Hmat_CSFs(list_UCSF_space,Enuc,obt,tbt)
            ndim_UCSF = len(list_ex_space) + len(list_pair_ex_space[1:])
            Hmat_UCSF = np.zeros([ndim_UCSF,ndim_UCSF])
            Hmat_UCSF[0:len(list_ex_space),0:len(list_ex_space)] = Hmat_ex
            list_pair_ex_space_cut = list_pair_ex_space[1:]
            for ii, CSF_new in enumerate(list_pair_ex_space_cut):
                for iCSF_orig,CSF_orig in enumerate(list_ex_space):
                    Helm = Helm_between_CSFs(Enuc,obt,tbt,CSF_orig,CSF_new)
                    Hmat_UCSF[iCSF_orig,len(list_ex_space)+ ii] = Helm
                    Hmat_UCSF[len(list_ex_space)+ ii,iCSF_orig] = Helm
                for jj in range(ii,len(list_pair_ex_space_cut)):
                    CSF_new_j = list_pair_ex_space_cut[jj]
                    Helm = Helm_between_CSFs(Enuc,obt,tbt,CSF_new,CSF_new_j)
                    Hmat_UCSF[len(list_ex_space)+ ii,len(list_ex_space)+ jj] = Helm
                    Hmat_UCSF[len(list_ex_space)+ jj,len(list_ex_space)+ ii] = Helm
            Hmat_UCSF_sparse = csr_matrix(Hmat_UCSF)
            E0_UCSF, psi_GS_UCSF = get_ground_state(Hmat_UCSF_sparse)
            lowering = E0_UCSF-E0_ex
            if lowering > 1e-8:
                print(f'Strange! Positive lowering in select_ia_pairs_for_CSF_E0: {lowering}. Bombing out!')
                sys.exit()
            if abs(lowering) > Ethrsh:
                list_ia_one_CSF.append(ia_pair)
                list_improve.append(lowering)
                print(f'CSF{iCSF},{ia_pair},E lowering: {lowering}')
#sorted_ampld_list, sorted_ia_pair_list,sorted_Ecorr_list  = zip(*sorted(zip(mp2_ampld_list,ia_pair_list,mp2_Ecorr_list)))

        if len(list_improve) != 0:
            sorted_list_improve, sorted_list_ia_one_CSF = zip(*sorted(zip(list_improve,list_ia_one_CSF)))
            print(f'sorted_list_improve: {sorted_list_improve}')
            sorted_list_ia_one_CSF = list(sorted_list_ia_one_CSF)
            list_list_ia_included[iCSF] += sorted_list_ia_one_CSF
            print(f'\nia_pairs included for CSF{iCSF}')
            print(list_list_ia_included[iCSF])

           #Update the list_ex_space with the new ex CSFs of the newly included ia_pairs
           #Very time consuming.
            list_theta = [0.0] * len(list_list_ia_included[iCSF])
            _, _, list_pair_ex_space, _, list_genmat, = \
              prepare_UCSF_for_one_CSF(CSF,list_list_ia_included[iCSF],list_theta,False)
            print(f'Done preparing list_ex_space for CSF{iCSF}')
           #list_ex_space += list_pair_ex_space[1:]
           #Hmat_ex = construct_Hmat_CSFs(list_ex_space,Enuc,obt,tbt)
           #Hmat_ex_sparse = csr_matrix(Hmat_ex)
           #E0_ex_new, _ = get_ground_state(Hmat_ex_sparse)
           #print(f'\nE0 of ex space with the expansion after handling CSF{iCSF}: {E0_ex_new}, lowering: {E0_ex_new - E0_ex}')
           #print(f'Dimension of updated ex space: {len(list_ex_space)}')
           #E0_ex = E0_ex_new #Set the reference for improvement for the next iCSF
            list_list_genmat.append(list_genmat)
            list_list_ex_space.append(list_pair_ex_space)
        else:
            list_list_genmat.append([])
            list_list_ex_space.append([list_CSF[iCSF]])

       #print(f'\nlist_list_ex_space[{iCSF}]: {list_list_ex_space[iCSF]}')


    return list_list_ex_space, list_list_ia_included, list_list_genmat

def get_ext_ampld_1by1(list_CSF,list_SOMO_DMO,list_sym_sign,list_list_ia_internal,list_list_ia_external,l_axial_sym,nparal=1,debug=False):

    if debug: print(f'\nIn get_ext_ampld_1by1')

    n_CSF = len(list_CSF)
   #First, generate genmats of internal ia pairs
    iref_group = -1
    list_list_genmat_int = []
    list_list_ex_space_int = []
    list_sym_CSC_vec_int = []
    for iCSF in range(n_CSF):
        if iCSF > 0 and l_axial_sym and abs(list_sym_sign[iCSF-1]) == 1: continue #This CSF has been considered as sym partner
        iref_group += 1
        list_ia_int = list_list_ia_internal[iref_group]
        list_zero_theta = [0.0]*len(list_ia_int)
        if l_axial_sym and list_sym_sign[iCSF] != 0:
            print(f'Making internal list_genmat for degenerate CSF{iCSF} and CSF{iCSF+1}')
            CSF1 = list_CSF[iCSF]
            CSF2 = list_CSF[iCSF+1]
            SOMO_DMO1 = list_SOMO_DMO[iCSF]
            SOMO_DMO2 = list_SOMO_DMO[iCSF+1]
            list_regroupped_ia, list_genmat, list_pair_ex_space, sym_CSF_vec = make_list_genmat_for_sym_CSF(CSF1,CSF2,SOMO_DMO1,SOMO_DMO2,list_sym_sign[iCSF],list_ia_int,list_zero_theta)
            print(f'list_ia_int: {list_ia_int} vs list_regroupped_ia: {list_regroupped_ia}')
           #The pairs in list_ia_int and list_regroupped_ia may not be identical. It is OK for the excitations between deg. shells.
        else:
            print(f'Making internal list_genmat for CSF{iCSF}')
            CSF = list_CSF[iCSF]
            UCSF, Uvec, list_pair_ex_space, Umat, list_genmat = \
              prepare_UCSF_for_one_CSF(CSF,list_ia_int,list_zero_theta,nparal,False)
            sym_CSF_vec = np.zeros(len(list_pair_ex_space))
            sym_CSF_vec[0] = 1.0

        list_list_genmat_int.append(list_genmat)
        list_list_ex_space_int.append(list_pair_ex_space)
        list_sym_CSC_vec_int.append(sym_CSF_vec)

    assert len(list_list_genmat_int) == len(list_list_ex_space_int)
    assert len(list_list_genmat_int) == len(list_sym_CSC_vec_int)

def select_ia_pairs_for_CSF_E0_grad(list_CSF,group_ia_pairs,Enuc,obt,tbt,Gthrsh=1.0e-4,debug=False):
    """
    Select ia pairs for each CSF for their individual gradient of E0
    """

    if debug: print('\nIn select_ia_pairs_for_CSF_E0_grad')

    Hmat_CSF = construct_Hmat_CSFs(list_CSF,Enuc,obt,tbt)

    Hmat_CSF_sparse = csr_matrix(Hmat_CSF)
    E0, psi_GS = get_ground_state(Hmat_CSF)
    print(f'\nE0 of original CSF space: {E0}')
    nCSF = len(list_CSF)
    list_list_ia_included = []
    list_list_genmat = []
    list_list_ex_space = []
    print(group_ia_pairs)
    for iCSF, CSF in enumerate(list_CSF):
        print(f'Working on CSF{iCSF}')
        list_ia_one_CSF = []
        list_grad = []
        for ia_pair in group_ia_pairs:
            T_op = FermionOperator()
            for ipair, pair in enumerate(ia_pair):
                T_op += get_Tiiaa_00(pair[1],pair[0])
            T_op = normal_ordered(T_op)
            T_op.compress()
           #print(ia_pair)
           #print(T_op)
            onlist, onidx_list, on_coefs = op_action_tz_remove_0coef(T_op,CSF[0],CSF[1],CSF[2])
            GCSF = [onlist,onidx_list,on_coefs]
            dE0_dtheta = 0.0
            for jCSF, CSFj in enumerate(list_CSF):
                HGelm = Helm_between_CSFs(Enuc,obt,tbt,CSFj,GCSF)
                dE0_dtheta += psi_GS[iCSF]*psi_GS[jCSF]*HGelm

            dE0_dtheta *= 2.0

            if abs(dE0_dtheta) >= Gthrsh:
                list_ia_one_CSF.append(ia_pair)
                list_grad.append(abs(dE0_dtheta))
               #print(f'CSF{iCSF}, {ia_pair}, {dE0_dtheta}')

        if len(list_grad) != 0:
            sorted_list_grad, sorted_list_ia_one_CSF = zip(*sorted(zip(list_grad,list_ia_one_CSF),reverse=True))
            print(f'sorted_list_grad: {sorted_list_grad}')
            sorted_list_ia_one_CSF = list(sorted_list_ia_one_CSF)
            list_list_ia_included.append(sorted_list_ia_one_CSF)
            print(f'\nia_pairs included for CSF{iCSF}')
            print(list_list_ia_included[iCSF])

            list_theta = [0.0] * len(list_list_ia_included[iCSF])
            _, _, list_pair_ex_space, _, list_genmat, = \
              prepare_UCSF_for_one_CSF(CSF,list_list_ia_included[iCSF],list_theta,False)
            print(f'Done preparing list_ex_space for CSF{iCSF}')
            list_list_genmat.append(list_genmat)
            list_list_ex_space.append(list_pair_ex_space)
        else:
            list_list_genmat.append([])
            list_list_ex_space.append([list_CSF[iCSF]])


#   return list_list_ex_space, list_list_ia_included, list_list_genmat

#def select_ia_pairs_for_CSF_E0_with_sym_CSF(list_CSF,list_SOMO_DMO,grouped_ia_pairs,Enuc,obt,tbt,l_include_ia_in_CAS,actmo_start,actmo_end,l_axial_sym=False,list_sym_sign=[],Ethrsh=1.0e-3,list_mo_exclud=[],nparal=1,debug=False):
    """
    Select ia pairs for each CSF for their individual lowerings of the E0 of the CSF space. The CSFs may be paired
    with their symmetry partners
    """

    if debug: print('\nIn select_ia_pairs_for_CSF_E0_with_sym_CSF')
   #print(f'grouped_ia_pairs in select_ia_pairs_for_CSF_E0_with_sym_CSF: {grouped_ia_pairs}')
    print("n grouped_ia_pairs =", len(grouped_ia_pairs))
    print(grouped_ia_pairs)

    if nparal > 1:
        Hmat_CSF = construct_Hmat_CSFs_paral_triu(list_CSF,Enuc,obt,tbt,nparal)
    else:
        Hmat_CSF = construct_Hmat_CSFs(list_CSF,Enuc,obt,tbt)
    E0_CSF, psi_GS_CSF = get_ground_state(Hmat_CSF)
    if debug: print(f'E0 of original CSF space: {E0_CSF}')
    nCSF = len(list_CSF)
    list_list_genmat = []
    list_list_ia_pair = []
    list_list_ex_space = []
    list_sym_CSF_vec = []
    n_sym_CSF = 0
    for iCSF in range(nCSF):
        if iCSF > 0 and l_axial_sym and abs(list_sym_sign[iCSF-1]) == 1: continue #This CSF has been considered as sym partner
        list_ia_one_CSF = []
        list_improve = []
        n_sym_CSF += 1
        if l_axial_sym and list_sym_sign[iCSF] != 0:
            print(f'\nSelecting ia pairs for degenerate CSF{iCSF} and CSF{iCSF+1}')
            CSF1 = list_CSF[iCSF]
            CSF2 = list_CSF[iCSF+1]
            for ia_pair in grouped_ia_pairs:
                l_ia_exclud = False
                for pair in ia_pair:
                    if pair[0] in list_mo_exclud or pair[1] in list_mo_exclud:
                        l_ia_exclud = True
                if l_ia_exclud:
                    print(f'{ia_pair} contains mo in list_mo_exclud: {list_mo_exclud}. Next.')
                    continue
                UCSF1, Uvec1, list_pair_ex_space1, Umat1, _ = \
                  prepare_UCSF_for_one_CSF(CSF1,[ia_pair],[0.0],nparal,False)
                UCSF2, Uvec2, list_pair_ex_space2, Umat2, _ = \
                  prepare_UCSF_for_one_CSF(CSF2,[ia_pair],[0.0],nparal,False)
                if len(list_pair_ex_space1) != len(list_pair_ex_space2):
                    print(f'Strange, the two list_ex_spaces of sym pairs do not have the same dimension')
                    print(len(list_pair_ex_space1),len(list_pair_ex_space2),ia_pair)
                    print('Bombing out!')
                    sys.exit()
                list_pair_ex_combine, list_corspnd, _ = combine_two_list_ex_spaces(list_pair_ex_space1,list_pair_ex_space2,list_sym_sign[iCSF],False)
               #if len(list_pair_ex_combine) != 2*len(list_pair_ex_space1):
               #    print(f'ia_pair: {ia_pair}')
               #    print('list_pair_ex_space1:')
               #    print(list_pair_ex_space1)
               #    print('list_pair_ex_space2:')
               #    print(list_pair_ex_space2)
               #    print('list_pair_ex_combine:')
               #    print(list_pair_ex_combine)
               #    print(f'list_corspnd: {list_corspnd}')
               #    input("Press Enter to continue...")
               #   #sys.exit()
                if len(list_pair_ex_space1) == 1:
                    print(f'{ia_pair} does not create any excited CSFs. Now jumping to the next pair.')
                    continue #No ex CSFs. Jump to next ia_pair
                list_extra_CSF = list_pair_ex_space1[1:] + list_pair_ex_combine[len(list_pair_ex_space1)+1:]
                if len(list_extra_CSF) == 0:
                    print(f'No extra CSFs in excitation. It should not have got here. Bombing out!')
                    sys.exit()
                ndim_with_extra_CSF = nCSF + len(list_extra_CSF)
                Hmat_CSF_extra = np.zeros([ndim_with_extra_CSF,ndim_with_extra_CSF])
                Hmat_CSF_extra[0:nCSF,0:nCSF] = Hmat_CSF
                if nparal > 1:
                    Hmat_extra = construct_Hmat_CSFs_paral_triu(list_extra_CSF,Enuc,obt,tbt,nparal)
                else:
                    Hmat_extra = construct_Hmat_CSFs(list_extra_CSF,Enuc,obt,tbt)
                Hmat_CSF_extra[nCSF:,nCSF:] = Hmat_extra
                for iextra,CSFextra in enumerate(list_extra_CSF):
                    for jCSF, CSForig in enumerate(list_CSF):
                        Helm = Helm_between_CSFs(Enuc,obt,tbt,CSForig,CSFextra)
                        Hmat_CSF_extra[nCSF+iextra,jCSF] = Helm
                        Hmat_CSF_extra[jCSF,nCSF+iextra] = Helm
                E0_extra, _ = get_ground_state(Hmat_CSF_extra)
                improve = E0_extra - E0_CSF
                print("Change in E_0:", improve)
                if improve > 1.0e-8:
                    print(f'Strange! E0 is larger with the extra CSFs by {improve}. Bombing out!')
                    sys.exit()
                l_ia_in_AS = True
                for pair in ia_pair:
                    if pair[0] < actmo_start or pair[1] > actmo_end:
                        l_ia_in_AS = False
                        print("Non-active space energy improvement", abs(improve))

                if abs(improve) > Ethrsh or (l_include_ia_in_CAS and l_ia_in_AS):
                    print(f'{ia_pair} is included')
                    list_improve.append(improve)
                    list_ia_one_CSF.append(ia_pair)
                    if abs(improve) > Ethrsh and (l_include_ia_in_CAS and l_ia_in_AS):
                        print(f'{ia_pair} included because both requirement met')
                    if abs(improve) > Ethrsh == True and (l_include_ia_in_CAS and l_ia_in_AS) == False:
                        print(f'{ia_pair} included because improvement threshold met')
                    if l_include_ia_in_CAS and l_ia_in_AS:
                        print(f'{ia_pair} included because being in active space')
                if abs(improve) < Ethrsh and not(l_include_ia_in_CAS and l_ia_in_AS):
                    print(f'{ia_pair} not included because improvement threshold not met and not in active space')
        else:
            print(f'\nSelecting ia pairs for CSF{iCSF}')
            CSF = list_CSF[iCSF]
            for ia_pair in grouped_ia_pairs:
                print(" ")
                print("Consider the ia pair", ia_pair)
                l_ia_in_AS = True
                l_ia_exclud = False
                for pair in ia_pair:
                    if pair[0] < actmo_start or pair[1] > actmo_end:
                        print(pair[0], " < ", actmo_start)
                        print(pair[1], " > ", actmo_end)
                        l_ia_in_AS = False
                        print("Non-active space energy improvement", abs(improve))
                    if pair[0] in list_mo_exclud or pair[1] in list_mo_exclud:
                        l_ia_exclud = True
                if l_ia_exclud:
                    print(f'{ia_pair} contain mo in list_mo_exclud: {list_mo_exclud}. Next.')
                    continue
                UCSF, Uvec, list_pair_ex_space, Umat, _ = \
                  prepare_UCSF_for_one_CSF(CSF,[ia_pair],[0.0],nparal,False)
                if len(list_pair_ex_space) == 1:
                    print(f'{ia_pair} does not create any excited CSFs. Now jumping to the next pair.')
                    continue #No ex CSFs. Jump to next ia_pair
                list_extra_CSF = list_pair_ex_space[1:]
                ndim_with_extra_CSF = nCSF + len(list_extra_CSF)
                Hmat_CSF_extra = np.zeros([ndim_with_extra_CSF,ndim_with_extra_CSF])
                Hmat_CSF_extra[0:nCSF,0:nCSF] = Hmat_CSF
                if nparal > 1:
                    Hmat_extra = construct_Hmat_CSFs_paral_triu(list_extra_CSF,Enuc,obt,tbt,nparal)
                else:
                    Hmat_extra = construct_Hmat_CSFs(list_extra_CSF,Enuc,obt,tbt)
                Hmat_CSF_extra[nCSF:,nCSF:] = Hmat_extra
                for iextra,CSFextra in enumerate(list_extra_CSF):
                    for jCSF, CSForig in enumerate(list_CSF):
                        Helm = Helm_between_CSFs(Enuc,obt,tbt,CSForig,CSFextra)
                        Hmat_CSF_extra[nCSF+iextra,jCSF] = Helm
                        Hmat_CSF_extra[jCSF,nCSF+iextra] = Helm
                E0_extra, _ = get_ground_state(Hmat_CSF_extra)
                improve = E0_extra - E0_CSF
                print("Change in E_0:", improve)
                if improve > 1.0e-8:
                    print(f'Strange! E0 is larger with the extra CSFs by {improve}. Bombing out!')
                    sys.exit()
                if abs(improve) > Ethrsh or (l_include_ia_in_CAS and l_ia_in_AS):
                    print(Ethrsh)
                    print(abs(improve) - Ethrsh)
                    print(abs(improve) > Ethrsh)
                    print((l_include_ia_in_CAS and l_ia_in_AS))
                    print(f'{ia_pair} is included')
                    list_improve.append(improve)
                    list_ia_one_CSF.append(ia_pair)
                    if abs(improve) > Ethrsh and (l_include_ia_in_CAS and l_ia_in_AS):
                        print(f'{ia_pair} included because both requirement met')
                    if abs(improve) > Ethrsh == True and (l_include_ia_in_CAS and l_ia_in_AS) == False:
                        print(f'{ia_pair} included because improvement threshold met')
                    if l_include_ia_in_CAS and l_ia_in_AS:
                        print(f'{ia_pair} included because being in active space')
                if abs(improve) < Ethrsh and not(l_include_ia_in_CAS and l_ia_in_AS):
                    print(f'{ia_pair} not included because improvement threshold not met and not in active space')

        if len(list_ia_one_CSF) != 0:
            sorted_list_improve, sorted_list_ia_one_CSF = zip(*sorted(zip(list_improve,list_ia_one_CSF)))
            sorted_list_ia_one_CSF = list(sorted_list_ia_one_CSF)
            print('ia_pairs included:')
            for ii, ia_pairs in enumerate(sorted_list_ia_one_CSF):
                print(ia_pairs, sorted_list_improve[ii])
        else:
            sorted_list_ia_one_CSF = []

        list_zero_theta = [0.0] * len(sorted_list_ia_one_CSF)
        if l_axial_sym and list_sym_sign[iCSF] != 0:
            print(f'Making list_genmat for degenerate CSF{iCSF} and CSF{iCSF+1}')
            CSF1 = list_CSF[iCSF]
            CSF2 = list_CSF[iCSF+1]
            SOMO_DMO1 = list_SOMO_DMO[iCSF]
            SOMO_DMO2 = list_SOMO_DMO[iCSF+1]
            list_regroupped_ia, list_genmat, list_pair_ex_combine, sym_CSF_vec = make_list_genmat_for_sym_CSF(CSF1,CSF2,SOMO_DMO1,SOMO_DMO2,list_sym_sign[iCSF],sorted_list_ia_one_CSF,list_zero_theta,False)
           #sys.exit()
           #UCSF1, Uvec1, list_pair_ex_space1, Umat1, list_genmat1 = \
           #  prepare_UCSF_for_one_CSF(CSF1,sorted_list_ia_one_CSF,list_zero_theta,False)
           #UCSF2, Uvec2, list_pair_ex_space2, Umat2, list_genmat2 = \
           #  prepare_UCSF_for_one_CSF(CSF2,sorted_list_ia_one_CSF,list_zero_theta,False)
           #if len(list_pair_ex_space1) != len(list_pair_ex_space2):
           #    print(f'Strange, the two list_ex_spaces of sym pairs do not have the same dimension')
           #    print(len(list_pair_ex_space1),len(list_pair_ex_space2),ia_pair)
           #    print('Bombing out!')
           #    sys.exit()
           #list_pair_ex_combine, list_corspnd, sym_CSF_vec = combine_two_list_ex_spaces(list_pair_ex_space1,list_pair_ex_space2,list_sym_sign[iCSF],False)
           #list_genmat = combine_list_genmat(list_genmat1,list_genmat2,list_corspnd)
           #check_combined_list_genmat(CSF1,CSF2,list_sym_sign[iCSF],list_pair_ex_combine,list_genmat,sym_CSF_vec,sorted_list_ia_one_CSF)
            list_list_ex_space.append(list_pair_ex_combine)
            list_list_ia_pair.append(list_regroupped_ia)
        else:
            print(f'Making list_genmat for CSF{iCSF}, nparal: {nparal}')
            CSF = list_CSF[iCSF]
           #If [[6,8],[5,7]] in list_ia_one_CSF, [[6,7],[5,8]] shall also be included
           #suppl_list_ia_one_CSF = supplement_pipi_pair_ex(sorted_list_ia_one_CSF,grouped_ia_pairs,False)
           #sorted_list_ia_one_CSF = copy.deepcopy(suppl_list_ia_one_CSF)
            UCSF, Uvec, list_pair_ex_space, Umat, list_genmat = \
              prepare_UCSF_for_one_CSF(CSF,sorted_list_ia_one_CSF,list_zero_theta,nparal,False)
            list_list_ex_space.append(list_pair_ex_space)
            list_list_ia_pair.append(sorted_list_ia_one_CSF)
            sym_CSF_vec = np.zeros(len(list_pair_ex_space))
            sym_CSF_vec[0] = 1.0

        list_list_genmat.append(list_genmat)
        list_sym_CSF_vec.append(sym_CSF_vec)
       #print('list_genmat:')
       #for genmat in list_genmat:
       #    print(genmat)
        print(f'Updated length: {len(list_list_genmat)}')

    assert len(list_list_ex_space) == len(list_list_ia_pair)
    assert len(list_list_ia_pair) == len(list_list_genmat)
    assert len(list_list_genmat) == len(list_sym_CSF_vec)
    print(f'Number of sym-adapted CSF basis: {len(list_list_ex_space)}')

    return list_list_ex_space, list_list_ia_pair, list_list_genmat, list_sym_CSF_vec

def get_lowest_n_eigen(H_sparse, n, tol=1e-10, maxiter=None):
    """
    Compute the lowest n eigenvalues and eigenvectors of a Hermitian sparse matrix.
    """
    evals, evecs = scipy.sparse.linalg.eigsh(
        H_sparse,
        k=n,
        which='SA',   # Smallest Algebraic
        tol=tol,
        maxiter=maxiter
    )
    idx = evals.argsort()
    return evals[idx]

def get_lowest_n_eigvec(H_sparse, n, tol=1e-10, maxiter=None):
    """
    Compute the lowest n eigenvalues and eigenvectors of a Hermitian sparse matrix.
    """
    evals, evecs = scipy.sparse.linalg.eigsh(
    H_sparse,
    k=n,
    which='SA',
    tol=tol,
    maxiter=maxiter
    )
    idx = evals.argsort()
    return evecs[:, idx].T

def group_ia_by_degeneracy(all_ia_pairs, degenerate_partners):
    """
    Group ia_pairs so that degenerate partners (sharing the same degenerate orbital) are together.
    Returns list of groups; each group is a list of ia_pair objects.
    """
    visited = set()
    groups = []
    key_to_ia = {}
    for ia in all_ia_pairs:
        ia_key = (ia[0][0], ia[0][1]) if isinstance(ia[0], list) else (ia[0], ia[1])
        key_to_ia[ia_key] = ia
    for ia in all_ia_pairs:
        ia_key = (ia[0][0], ia[0][1]) if isinstance(ia[0], list) else (ia[0], ia[1])
        if ia_key in visited:
            continue
        visited.add(ia_key)
        i_orb, a_orb = ia_key
        group = [ia]
        i_partners = degenerate_partners.get(i_orb, [])
        a_partners = degenerate_partners.get(a_orb, [])
        if i_partners and a_partners:
            # Both i and a are degenerate: swap both
            for ip in i_partners:
                for ap in a_partners:
                    pk = (ip, ap)
                    if pk in key_to_ia and pk not in visited:
                        group.append(key_to_ia[pk])
                        visited.add(pk)
        elif not i_partners and a_partners:
            # i is non-degenerate (σ), a is degenerate: partner is (i, a')
            for ap in a_partners:
                pk = (i_orb, ap)
                if pk in key_to_ia and pk not in visited:
                    group.append(key_to_ia[pk])
                    visited.add(pk)
        elif i_partners and not a_partners:
            # a is non-degenerate (σ), i is degenerate: partner is (i', a)
            for ip in i_partners:
                pk = (ip, a_orb)
                if pk in key_to_ia and pk not in visited:
                    group.append(key_to_ia[pk])
                    visited.add(pk)
        # else: both non-degenerate, group stays as singleton with ΔML=0
        groups.append(group)
    return groups


def group_total_delta_ml(group, mo_ml):
    """Sum of ΔML = 2*(ml[a]-ml[i]) over all pairs in the group."""
    total = 0
    for ia in group:
        flat = ia if not isinstance(ia[0], list) else ia
        for pair in flat:
            total += 2 * (mo_ml[pair[1]] - mo_ml[pair[0]])
    return total


def chain_delta_ml(chain_pairs, mo_ml):
    return sum(2*(mo_ml[a] - mo_ml[i]) for i, a in chain_pairs)


def chain_needs_partner(chain_pairs, mo_ml, dinfh_degenerate_groups):
    group_counts = {tuple(g): {+1: 0, -1: 0} for g in dinfh_degenerate_groups}
    for i, a in chain_pairs:
        for group in dinfh_degenerate_groups:
            if i in group or a in group:
                comp = +1 if mo_ml.get(i, 0) > 0 or mo_ml.get(a, 0) > 0 else -1
                group_counts[tuple(group)][comp] += 1
    return any(v[+1] != v[-1] for v in group_counts.values())


def build_partner_chain(chain_pairs, degenerate_partners):
    return [(degenerate_partners.get(i, [i])[0],
             degenerate_partners.get(a, [a])[0])
            for i, a in chain_pairs]


def filter_CSFs_cascade(list_CSF_full, list_SOMO_DMO_original, seed_indices,
                        n_spatialmo, n_shells=3, max_hd=4,
                        degen_thrsh=1e-8, Enuc=None, obt=None, tbt=None):
    """
    Filter CSFs by symmetry using the Frontier-Coupling Cascade.

    A direct H-coupling check ⟨seed|H|CSF⟩ ≠ 0 fails for seniority-6+ CSFs
    because Slater-Condon makes the matrix element identically zero whenever the
    spin-orbital Hamming distance exceeds 4 (more than a double excitation).
    This function replaces that single-hop check with a multi-shell BFS:

      Shell 0   — seed CSFs (formula-reliable irrep, e.g. all-ml=0 A1g CSFs)
      Shell k+1 — all previously-unseen CSFs whose spatial-orbital occupation
                  Hamming distance from any Shell-k CSF is ≤ max_hd

    Because [H, sym_op] = 0, any CSF reachable from A1g via H belongs to A1g.
    Bitstring Hamming distance (graphical connectivity) is used instead of
    numerical H_ij to avoid false negatives from accidentally-zero integrals
    (e.g. a matrix element that is exactly zero due to an orbital node even
    though the symmetry connection is real).

    After the BFS the kept set is expanded to whole degenerate groups so that
    both components of every degenerate pair are always included together.

    Parameters
    ----------
    list_CSF_full          : full CSF list before filtering
    list_SOMO_DMO_original : companion [SOMO, DMO, ...] info
    seed_indices           : indices of reliably-labelled seed CSFs (e.g. A1g)
    n_spatialmo            : number of spatial orbitals
    n_shells               : BFS depth (3 covers singles→doubles→triples→quadruples
                             and therefore seniority-6 from a seniority-0 reference)
    max_hd                 : max spatial Hamming distance per BFS step (4 = doubles)
    degen_thrsh            : threshold for degenerate-group detection via E_diag
    Enuc, obt, tbt         : integrals for diagonal energy (degenerate group expansion);
                             if None the degenerate-group step is skipped

    Returns
    -------
    sorted list of int — indices into list_CSF_full that pass the cascade filter
    """
    n_full = len(list_CSF_full)

    # ── Spatial occupation vector for each CSF from SOMO/DMO info ────────────
    def _occ_vec(somo_dmo):
        somo, dmo = somo_dmo[0], somo_dmo[1]
        occ = np.zeros(n_spatialmo, dtype=np.int8)
        for i in dmo:
            if 0 <= i < n_spatialmo:
                occ[i] = 2
        for i in somo:
            if 0 <= i < n_spatialmo:
                occ[i] = 1
        return occ

    occ_vecs = [_occ_vec(sm) for sm in list_SOMO_DMO_original]

    # ── BFS expansion ─────────────────────────────────────────────────────────
    kept     = set(seed_indices)
    frontier = set(seed_indices)

    def _shell_label(j):
        """One-line descriptor for CSF j: index, label, ML, seniority, SOMO, DMO."""
        sm = list_SOMO_DMO_original[j]
        somo, dmo = sm[0], sm[1]
        label = sm[2] if len(sm) > 2 else '?'
        ml    = sm[3] if len(sm) > 3 else '?'
        par   = sm[4] if len(sm) > 4 else '?'
        sen   = len(somo)
        return (f'  idx={j:4d}  {label:6s}  ML={ml!s:4s}  par={par!s:2s}  '
                f'sen={sen}  SOMO={somo}  DMO={dmo}')

    for shell in range(n_shells):
        new_shell = set()
        for j in range(n_full):
            if j in kept:
                continue
            occ_j = occ_vecs[j]
            for f in frontier:
                hd = int(np.sum(occ_vecs[f] != occ_j))
                if hd <= max_hd:
                    new_shell.add(j)
                    break   # one match is enough to include j
        n_new = len(new_shell)
        print(f'  Cascade shell {shell + 1}: {n_new} new CSFs '
              f'(running total: {len(kept) + n_new})')
        for j in sorted(new_shell):
            print(_shell_label(j))
        if n_new == 0:
            break           # converged — no new CSFs reachable
        frontier = new_shell
        kept    |= new_shell

    # ── Expand to full degenerate groups ──────────────────────────────────────
    if Enuc is not None:
        E_diag   = [Helm_between_CSFs(Enuc, obt, tbt, c, c) for c in list_CSF_full]
        assigned = [False] * n_full
        degen_groups = []
        for i in range(n_full):
            if assigned[i]:
                continue
            grp = [i]
            for j in range(i + 1, n_full):
                if not assigned[j] and abs(E_diag[i] - E_diag[j]) < degen_thrsh:
                    grp.append(j)
            for k in grp:
                assigned[k] = True
            degen_groups.append(grp)

        include_set = set()
        for grp in degen_groups:
            if any(k in kept for k in grp):
                include_set.update(grp)
                if len(grp) > 1:
                    print(f'  Cascade: degenerate group {grp} kept whole '
                          f'(member(s) {[k for k in grp if k in kept]} reached by BFS)')
    else:
        include_set = kept

    result = sorted(include_set)
    print(f'  Cascade filter: {len(result)} CSFs retained out of {n_full} total '
          f'({n_shells} shells, max_hd={max_hd})')
    return result


def _compute_improve_for_group(degen_group, CSF, nCSF, Hmat_CSF, Enuc, obt, tbt,
                                list_CSF, eigen, evals_CSF, iCSF, nparal,
                                ref_csf_list=None):
    """Add all excitation CSFs from every member of the degenerate group
    into one augmented H matrix.
    Returns (improve, isign_list, isign_ref_by_iapair):
      - improve: MEAN energy lowering over the targeted states (any number).
      - isign_list[k]: sign of the k-th extra CSF relative to the FIRST extra
        CSF in the lowest eigenvector (for combining degenerate excitation pairs).
      - isign_ref_by_iapair[key]: sign of that ia_pair's excitation relative to
        the REFERENCE CSF in the lowest eigenvector (for folding a reference that
        is degenerate with its own excitation). Both use the same ground-state
        eigenvector, analogous to l_axial_sym's sign(psi[i])*sign(psi[j]).
    ref_csf_list: if given, each ia_pair is applied to every CSF in this list
        (not just CSF) so that ia_pairs firing in reverse on one degenerate
        reference member but forward on another are not missed."""
    # When the reference is a degenerate group, apply each ia_pair to ALL members
    # so that reverse-fire (de-excitation) on one member doesn't hide a genuine
    # forward excitation on another member.
    _csfs_for_gen = ref_csf_list if ref_csf_list is not None else [CSF]
    all_extra_CSFs = []
    extra_offset_by_iapair = []   # (ia_pair, start offset of its extras in all_extra_CSFs)
    for ia_pair in degen_group:
        if ia_pair == 'None':
            continue  # reference itself — already in Hmat_CSF, not an extra CSF
        first_off_for_pair = None
        _seen_for_pair = set()   # deduplicate by SD-index frozenset within this ia_pair
        for CSF_r in _csfs_for_gen:
            _, _, list_pair_ex_space_r, _, _ = \
                prepare_UCSF_for_one_CSF(CSF_r, [ia_pair], [0.0], nparal, False)
            for ex_csf in list_pair_ex_space_r[1:]:
                ex_key = frozenset(int(x) for x in ex_csf[1])
                if ex_key in _seen_for_pair:
                    continue
                _seen_for_pair.add(ex_key)
                if first_off_for_pair is None:
                    first_off_for_pair = len(all_extra_CSFs)
                all_extra_CSFs.append(ex_csf)
        if first_off_for_pair is not None:
            extra_offset_by_iapair.append((ia_pair, first_off_for_pair))
    if len(all_extra_CSFs) == 0:
        return None, [], {}
    ndim_aug = nCSF + len(all_extra_CSFs)
    Hmat_aug = np.zeros([ndim_aug, ndim_aug])
    Hmat_aug[0:nCSF, 0:nCSF] = Hmat_CSF
    if nparal > 1:
        Hmat_aug[nCSF:, nCSF:] = construct_Hmat_CSFs_paral_triu(all_extra_CSFs, Enuc, obt, tbt, nparal)
    else:
        Hmat_aug[nCSF:, nCSF:] = construct_Hmat_CSFs(all_extra_CSFs, Enuc, obt, tbt)
    for iextra, CSFextra in enumerate(all_extra_CSFs):
        for jCSF, CSForig in enumerate(list_CSF):
            Helm = Helm_between_CSFs(Enuc, obt, tbt, CSForig, CSFextra)
            Hmat_aug[nCSF + iextra, jCSF] = Helm
            Hmat_aug[jCSF, nCSF + iextra] = Helm
    # Get eigenvalues AND eigenvectors (mirrors l_axial_sym's use of psi_GS)
    Hmat_aug_sparse = csr_matrix(Hmat_aug)
    n_eig = min(eigen, ndim_aug - 1)
    evals_aug, evecs_aug = scipy.sparse.linalg.eigsh(Hmat_aug_sparse, k=n_eig, which='SA')
    idx_sort = np.argsort(evals_aug)
    evals_aug = evals_aug[idx_sort]
    evecs_aug = evecs_aug[:, idx_sort]
    # Lowest eigenvector: read signs of extra-CSF components
    # (analogous to sign(psi[iCSF]) * sign(psi[iCSF+1]) in l_axial_sym)
    psi_gs = evecs_aug[:, 0]
    sign_ref = np.sign(psi_gs[nCSF]) if abs(psi_gs[nCSF]) > 1e-10 else 1.0
    isign_list = []
    for k in range(len(all_extra_CSFs)):
        s = np.sign(psi_gs[nCSF + k]) if abs(psi_gs[nCSF + k]) > 1e-10 else sign_ref
        isign_list.append(round(sign_ref * s))
    # Reference-relative signs (same GS-eigenvector mechanism, but relative to the
    # reference CSF, which sits at index iCSF in the Hmat_CSF block of the augmented H):
    #   isign = sign(psi[ref]) * sign(psi[T_ia ref])
    # Used when the reference itself is degenerate with an excitation ('None' group),
    # so the sign is taken consistently from the ground state, not a 2-state shortcut.
    sign_ref_csf = np.sign(psi_gs[iCSF]) if abs(psi_gs[iCSF]) > 1e-10 else 1.0
    print(f'  [isign] refCSF at idx {iCSF}: psi={psi_gs[iCSF]:.6f}  sign={sign_ref_csf:+.0f}')
    isign_ref_by_iapair = {}
    for (ia_pair_off, off) in extra_offset_by_iapair:
        s = np.sign(psi_gs[nCSF + off]) if abs(psi_gs[nCSF + off]) > 1e-10 else sign_ref_csf
        key = tuple(tuple(p) if isinstance(p, list) else p for p in ia_pair_off)
        isign_ref_by_iapair[key] = round(sign_ref_csf * s)
        print(f'  [isign] T_{ia_pair_off}|refCSF> at idx {nCSF+off}: '
              f'psi={psi_gs[nCSF+off]:.6f}  sign={s:+.0f}  '
              f'isign=sign_ref*sign_ex={round(sign_ref_csf*s):+d}')
    evals_aug_arr = np.array(sorted(evals_aug))[:eigen]
    # match lengths so eigen==1 (or a tiny augmented space) is safe
    evals_CSF_arr = np.array(evals_CSF)[:len(evals_aug_arr)]
    # rely ONLY on the mean improvement over the targeted states
    improve    = (evals_aug_arr - evals_CSF_arr).mean()
    return improve, isign_list, isign_ref_by_iapair


def _compute_improve_for_pair(CSF, ia_pair, nparal, nCSF, Hmat_CSF, Enuc, obt, tbt,
                               list_CSF, eigen, evals_CSF):
    UCSF, Uvec, list_pair_ex_space, Umat, _ = \
        prepare_UCSF_for_one_CSF(CSF, [ia_pair], [0.0], nparal, False)
    if len(list_pair_ex_space) == 1:
        return None
    list_extra_CSF = list_pair_ex_space[1:]
    ndim_with_extra_CSF = nCSF + len(list_extra_CSF)
    Hmat_CSF_extra = np.zeros([ndim_with_extra_CSF, ndim_with_extra_CSF])
    Hmat_CSF_extra[0:nCSF, 0:nCSF] = Hmat_CSF
    if nparal > 1:
        Hmat_extra = construct_Hmat_CSFs_paral_triu(list_extra_CSF, Enuc, obt, tbt, nparal)
    else:
        Hmat_extra = construct_Hmat_CSFs(list_extra_CSF, Enuc, obt, tbt)
    Hmat_CSF_extra[nCSF:, nCSF:] = Hmat_extra
    for iextra, CSFextra in enumerate(list_extra_CSF):
        if nparal > 1:
            helm_row = Parallel(n_jobs=nparal)(
                delayed(Helm_between_CSFs)(Enuc, obt, tbt, CSForig, CSFextra)
                for CSForig in list_CSF)
        else:
            helm_row = [Helm_between_CSFs(Enuc, obt, tbt, CSForig, CSFextra)
                        for CSForig in list_CSF]
        for jCSF, Helm in enumerate(helm_row):
            Hmat_CSF_extra[nCSF + iextra, jCSF] = Helm
            Hmat_CSF_extra[jCSF, nCSF + iextra] = Helm
    evals_extra = get_lowest_n_eigen(Hmat_CSF_extra, eigen)
    diffs = evals_extra - evals_CSF
    improve = diffs.mean()
    return improve


# ── Helpers shared by select_ia_pairs_for_CSF_Eavg_with_sym_CSF ──────────────

def ia_hashable_key(ia):
    """Return a hashable key for an ia-pair (handles plain, group, and effective-pair formats)."""
    return tuple(tuple(p) if isinstance(p, list) else (p,) for p in ia)


def is_effective_pair(item):
    """True if item is a sym-adapted effective pair: list of (coeff, ia_group) tuples."""
    return (isinstance(item, list) and len(item) > 0
            and isinstance(item[0], tuple)
            and isinstance(item[0][0], float))


def apply_single_excitation(sds, coefs, dmo, vmo):
    """Apply a single T_dmo→vmo excitation at the raw SD level."""
    dmoa, dmob, vmoa, vmob = 2*dmo, 2*dmo+1, 2*vmo, 2*vmo+1
    out_sds, out_coefs = [], []
    for ii, ov in enumerate(sds):
        l_fwd = (ov[dmoa]==1. and ov[dmob]==1. and ov[vmoa]==0. and ov[vmob]==0.)
        l_rev = (ov[dmoa]==0. and ov[dmob]==0. and ov[vmoa]==1. and ov[vmob]==1.)
        if not l_fwd and not l_rev:
            continue
        new_ov = np.array(ov, dtype=float)
        if l_fwd:
            new_ov[dmoa], new_ov[dmob], new_ov[vmoa], new_ov[vmob] = 0, 0, 1, 1
            sgn = 1.0
        else:
            new_ov[vmoa], new_ov[vmob], new_ov[dmoa], new_ov[dmob] = 0, 0, 1, 1
            sgn = -1.0
        out_sds.append(new_ov)
        out_coefs.append(float(coefs[ii]) * sgn)
    return (out_sds, np.array(out_coefs)) if out_sds else (None, None)


def apply_excitation_group(sds, coefs, group):
    """Apply a sequence of excitations (group of pairs) in order."""
    cur_sds, cur_coefs = list(sds), np.array(coefs)
    for pair in group:
        cur_sds, cur_coefs = apply_single_excitation(cur_sds, cur_coefs, pair[0], pair[1])
        if cur_sds is None:
            return None, None
    return cur_sds, cur_coefs


def apply_effective_pair(sds, coefs, item):
    """Apply a plain pair, group pair, or sym-adapted effective pair to (sds, coefs)."""
    if is_effective_pair(item):
        all_sds, all_coefs = [], []
        for c_k, ia_group in item:
            ns, nc = apply_excitation_group(sds, coefs, ia_group)
            if ns is not None:
                all_sds.extend(ns)
                all_coefs.extend((np.array(nc) * float(c_k)).tolist())
        return (all_sds, np.array(all_coefs)) if all_sds else (None, None)
    elif isinstance(item[0], list):
        return apply_excitation_group(sds, coefs, item)
    else:
        return apply_single_excitation(sds, coefs, item[0], item[1])


def canonicalize_chain_state(sds, coefs):
    """Merge duplicate SDs, normalize, and return (sds, sd_indices, coefs) or (None,None,None)."""
    from collections import defaultdict
    merged = defaultdict(float)
    sd_by_key = {}
    for ov, c in zip(sds, coefs):
        k = tuple(int(x) for x in ov)
        merged[k] += float(c)
        sd_by_key[k] = ov
    kept_keys = [k for k, c in merged.items() if abs(c) > 1e-12]
    if not kept_keys:
        return None, None, None
    final_sds = [sd_by_key[k] for k in kept_keys]
    final_coefs = np.array([merged[k] for k in kept_keys])
    norm = np.sqrt(np.dot(final_coefs, final_coefs))
    if norm < 1e-12:
        return None, None, None
    final_idx = [get_on_idx(np.array(sd)) for sd in final_sds]
    return final_sds, final_idx, final_coefs / norm


def remap_genmat_to_combined(G_local, local_to_combined_map, combined_dim):
    """Remap a local-space sparse genmat into a combined-space sparse matrix."""
    G_out = csr_matrix((combined_dim, combined_dim), dtype=float)
    if not scipy.sparse.issparse(G_local):
        G_local = csr_matrix(np.asarray(G_local, dtype=float))
    r, c = G_local.nonzero()
    for ii in range(len(r)):
        G_out[local_to_combined_map[r[ii]], local_to_combined_map[c[ii]]] = G_local[r[ii], c[ii]]
    return G_out

from collections import defaultdict as _dd
def build_chain_group_excitation_space(chain_csf, sub_pairs, sub_map, n_eff, nparal):
    """Build (ex_space, genmat) for one chain basis group.

    Applies each sub-generator from sub_pairs to chain_csf via op_action_tz_CSF,
    GS-orthogonalizes in SD-index space, then recombines sub-pair genmats into
    effective-pair genmats using sub_map.  Works for multi-SD chain states with
    mixed spatial occupancies (bypasses make_pair_ex_space_manual's occupancy check).
    """
    gs_tol = 1e-8
    # raw_states = [chain_csf]
    # _eff_groups = _dd(list)
    # for _k, (_ei, _co) in enumerate(sub_map):
    #     _eff_groups[_ei].append((sub_pairs[_k], _co))
    # for _ei in sorted(_eff_groups.keys()):
    #     _combined = {}
    #     for _sp, _co in _eff_groups[_ei]:
    #         _T = get_Tiiaa_00(_sp[0], _sp[1])
    #         _st = op_action_tz_CSF(_T, chain_csf)
    #         if _st is None:
    #             continue
    #         for _ov, _idx, _c in zip(_st[0], _st[1], _st[2]):
    #             _key = int(_idx)
    #             if _key in _combined:
    #                 _combined[_key] = (_ov, _combined[_key][1] + float(_c) * _co)
    #             else:
    #                 _combined[_key] = (_ov, float(_c) * _co)
    #     _combined = {k: v for k, v in _combined.items() if abs(v[1]) > 1e-12}
    #     if not _combined:
    #         continue
    #     _idxs  = sorted(_combined.keys())
    #     _sds   = [_combined[k][0] for k in _idxs]
    #     _coefs = np.array([_combined[k][1] for k in _idxs])
    #     _norm  = np.linalg.norm(_coefs)
    #     if _norm < gs_tol:
    #         continue
    #     raw_states.append([_sds, _idxs, _coefs / _norm])

    raw_states = [chain_csf]

    seen_sd = set()

    for _sp in sub_pairs:

        T = get_Tiiaa_00(_sp[0], _sp[1])
        st = op_action_tz_CSF(T, chain_csf)

        if st is None:
            continue

        sds, idxs, coefs = st

        for sd_vec, sd_idx, coef in zip(sds, idxs, coefs):

            sd_idx = int(sd_idx)

            # skip duplicates
            if sd_idx in seen_sd:
                continue
            seen_sd.add(sd_idx)

            # each determinant becomes its own basis vector
            raw_states.append([
                [sd_vec],  # single SD only
                [sd_idx],  # single index
                np.array([1.0])  # normalized SD basis
            ])

    sd_map = {}
    for s in raw_states:
        for sd, si in zip(s[0], s[1]):
            sd_map[int(si)] = sd
    Q_states = []

    for st in raw_states:
        v = {int(k): float(c) for k, c in zip(st[1], st[2])}

        n = np.sqrt(sum(c ** 2 for c in v.values()))
        if n < 1e-16:
            continue

        v = {k: c / n for k, c in v.items()}

        idx = list(v.keys())
        Q_states.append([[sd_map[k] for k in idx],idx,np.array([v[k] for k in idx])])
    ex_sp = Q_states
    print(sub_pairs)
    sub_gm = make_iapair_genmat_new(sub_pairs, ex_sp, nparal, debug = False)
    n_b = len(ex_sp)
    comb = [None] * n_eff
    for k, (ei, co) in enumerate(sub_map):
        g = sub_gm[k] * co
        comb[ei] = g if comb[ei] is None else comb[ei] + g
    gm = [g if g is not None else csr_matrix((n_b, n_b)) for g in comb]
    return ex_sp, gm


def build_combined_chain_excitation_space(chain_csfs, sub_pairs, sub_map, n_eff, nparal):
    """Build ONE shared rotation space + genmat for ALL chain CSFs of a refCSF.

    The space is spanned by *individual SDs*: the union over every chain CSF of
    (i) the chain determinants themselves and (ii) the determinants produced by
    applying each leftover sub-generator T_p to that chain CSF.  A single set of
    effective-pair generator matrices is built on this common orthonormal SD
    basis, so exp(Σ θ_p G_p) is ONE unitary U shared by every chain group.

    Because U is unitary on the common space, W|chain_a> and W|chain_b> remain
    orthogonal whenever the reference chain states are orthogonal — so the
    basis-extension UCSFs are orthogonal by construction (no metric artifact).

    Returns
    -------
    ex_sp      : common orthonormal SD basis ( list of [ [sd], [idx], [1.0] ] )
    gm         : list of n_eff effective-pair genmats acting on ex_sp
    chain_vecs : list parallel to chain_csfs; chain_vecs[k] is that chain CSF
                 expressed as a normalized coefficient vector over ex_sp
    """
    # 1. Union of all SD indices: chain determinants + every T_p|chain>.
    sd_map = {}                       # idx -> sd_vec
    for chain_csf in chain_csfs:
        for sd, si in zip(chain_csf[0], chain_csf[1]):
            sd_map[int(si)] = sd
        for _sp in sub_pairs:
            T = get_Tiiaa_00(_sp[0], _sp[1])
            st = op_action_tz_CSF(T, chain_csf)
            if st is None:
                continue
            for sd, si, c in zip(st[0], st[1], st[2]):
                sd_map[int(si)] = sd

    # 2. Orthonormal SD basis: one unit basis vector per unique determinant.
    idx_order = sorted(sd_map.keys())
    idx_pos = {idx: p for p, idx in enumerate(idx_order)}
    ex_sp = [[[sd_map[idx]], [idx], np.array([1.0])] for idx in idx_order]

    # 3. Express each chain CSF as a normalized coefficient vector over ex_sp.
    chain_vecs = []
    for chain_csf in chain_csfs:
        v = np.zeros(len(idx_order))
        for si, c in zip(chain_csf[1], chain_csf[2]):
            v[idx_pos[int(si)]] += float(c)
        nrm = np.linalg.norm(v)
        if nrm > 1e-16:
            v = v / nrm
        chain_vecs.append(v)

    # 4. ONE set of genmats on the common SD basis, folded into effective pairs.
    sub_gm = make_iapair_genmat_new(sub_pairs, ex_sp, nparal, debug=False)
    n_b = len(ex_sp)
    comb = [None] * n_eff
    for k, (ei, co) in enumerate(sub_map):
        g = sub_gm[k] * co
        comb[ei] = g if comb[ei] is None else comb[ei] + g
    gm = [g if g is not None else csr_matrix((n_b, n_b)) for g in comb]
    return ex_sp, gm, chain_vecs


def build_csf_to_ref_group(Hmat_CSF, nCSF, mo_ml):
    """Group CSF indices by equal diagonal energy (degenerate reference shells).

    Returns csf_to_ref_grp: dict mapping each CSF index to its degenerate group list.
    Returns an empty dict when mo_ml is None (no symmetry grouping needed).
    """
    if mo_ml is None:
        return {}
    E_diag = [Hmat_CSF[i, i] for i in range(nCSF)]
    degen_thrsh = 1e-8
    visited = set()
    groups = []
    for i in range(nCSF):
        if i in visited:
            continue
        visited.add(i)
        grp = [i]
        for j in range(i + 1, nCSF):
            if j not in visited and abs(E_diag[i] - E_diag[j]) < degen_thrsh:
                grp.append(j)
                visited.add(j)
        groups.append(grp)
    csf_to_ref_grp = {ci: grp for grp in groups for ci in grp}
    print('\nReference CSF degenerate groups (by diagonal energy):')
    for grp in groups:
        if len(grp) > 1:
            print(f'  Group {grp}  E={E_diag[grp[0]]:.8f}')
    return csf_to_ref_grp


def apply_item_to_CSF(ia_item, T_state_in):
    """Apply an ia_item (plain pair, group, or effective pair) to a full CSF state object."""
    if is_effective_pair(ia_item):
        parts = []
        for c_k, ia_group in ia_item:
            T_sub = T_state_in
            for pair in ia_group:
                T_op_k = get_Tiiaa_00(pair[0], pair[1])
                T_sub = op_action_tz_CSF(T_op_k, T_sub)
            norm_k = np.sqrt(overlap_CSFs(T_sub, T_sub))
            if norm_k > 1e-9:
                parts.append((float(c_k), T_sub))
        if not parts:
            return None
        merged = {}
        for c_k, (onl, idx, coef) in parts:
            for on, ix, co in zip(onl, idx, coef):
                k = tuple(int(x) for x in np.atleast_1d(on))
                prev = merged.get(k, (on, ix, 0.0))
                merged[k] = (prev[0], prev[1], prev[2] + float(co) * c_k)
        kept = [(on, ix, co) for on, ix, co in merged.values() if abs(co) > 1e-12]
        if not kept:
            return None
        f_coef = np.array([co for _, _, co in kept])
        norm = np.sqrt(np.dot(f_coef, f_coef))
        if norm < 1e-12:
            return None
        return [[on for on, _, _ in kept], [ix for _, ix, _ in kept], f_coef / norm]
    elif isinstance(ia_item[0], list):
        T_state = T_state_in
        for pair in ia_item:
            T_op_k = get_Tiiaa_00(pair[0], pair[1])
            T_state = op_action_tz_CSF(T_op_k, T_state)
        norm = np.sqrt(overlap_CSFs(T_state, T_state))
        if norm < 1e-9:
            return None
        return [T_state[0], T_state[1], T_state[2] / norm]
    else:
        print(f'  [WARNING] apply_item_to_CSF: unrecognized format {ia_item}')
        return None

def select_ia_pairs_sym_aware(CSF, iCSF, grouped_ia_pairs, Enuc, obt, tbt, nparal,
        l_include_ia_in_CAS, actmo_start, actmo_end, ratio, Ethrsh, list_mo_exclud,
        nCSF, Hmat_CSF, list_CSF, eigen, evals_CSF,csf_to_ref_grp, list_improvement, list_improvement_ia):
    """Symmetry-aware (D∞h) ia-pair selection for one CSF.

    Returns (list_ia_one_CSF, list_improve, list_ia_basis_one_CSF, list_basis_improve,
             list_selected_sgen_groups, list_ref_degenerate_groups).
    """
    list_ia_one_CSF = []
    list_ia_basis_one_CSF = []
    list_improve = []
    list_basis_improve = []
    degen_thrsh = 1e-8
    excitation_candidates = []
    ia_ex_csf = {}
    ref_grp_sel = csf_to_ref_grp.get(iCSF, [iCSF])
    for ia_pair in grouped_ia_pairs:
        pair_is_excluded = False
        pair_in_active_space = True
        for _pair in ia_pair:
            if _pair[0] in list_mo_exclud or _pair[1] in list_mo_exclud:
                pair_is_excluded = True
            if _pair[0] < actmo_start or _pair[1] > actmo_end:
                pair_in_active_space = False
        if pair_is_excluded:
            print(f'  [degen-path] {ia_pair} contains excluded orbital '
                  f'(list_mo_exclud={list_mo_exclud}), skipping')
            continue
        if not pair_in_active_space and not l_include_ia_in_CAS:
            print(f'  [degen-path] {ia_pair} outside active space '
                  f'[{actmo_start},{actmo_end}], skipping')
            continue

        any_ex = False
        _seen_ex_for_pair = set()   # deduplicate within this ia_pair
        for ci in ref_grp_sel:
            _, _, ex_space_ci, _, _ = prepare_UCSF_for_one_CSF(
                list_CSF[ci], [ia_pair], [0.0], nparal, False)
            for ex_csf_ci in ex_space_ci[1:]:
                ex_key = frozenset(int(x) for x in ex_csf_ci[1])
                if ex_key in _seen_ex_for_pair:
                    continue
                _seen_ex_for_pair.add(ex_key)
                E_ex_ci = Helm_between_CSFs(Enuc, obt, tbt, ex_csf_ci, ex_csf_ci)
                excitation_candidates.append((ia_pair, ex_csf_ci, E_ex_ci))
                key = tuple(tuple(p) if isinstance(p, list) else p for p in ia_pair)
                if key not in ia_ex_csf:
                    ia_ex_csf[key] = ex_csf_ci
                print(f'{ia_pair} (from ref CSF{ci})  E_ex={E_ex_ci:.10f}')
                any_ex = True
        if not any_ex:
            print(f'{ia_pair} produces no excited CSF from any member of '
                  f'ref group {ref_grp_sel}. Skipping.')
    E_ref = Helm_between_CSFs(Enuc, obt, tbt, list_CSF[iCSF], list_CSF[iCSF])
    print(f'Reference CSF diagonal energy: {E_ref}')
    excitation_candidates.append((None, list_CSF[iCSF], E_ref))
    excitation_candidates = excitation_candidates[::-1]
    print(f'Collected refCSF excitations: {excitation_candidates}')

    print(f'\n=== E_avg diagnostic for CSF{iCSF} (degen_thrsh={degen_thrsh:.1e}) ===')
    print(f'  {"ia_pair":<30} {"E_ex":>14}  {"ΔE_from_ref":>14}  status')
    print(f'  {"":->30}-{"":->14}--{"":->14}--{"":->20}')
    _e_ref_diag = None
    for _ia_p, _ex_c, _E in excitation_candidates:
        if _ia_p is None:
            _e_ref_diag = _E
            print(f'  {"[ref CSF]":<30} {_E:>14.10f}  {"---":>14}  (reference)')
        else:
            _dE = _E - _e_ref_diag if _e_ref_diag is not None else float('nan')
            _close = abs(_dE) < degen_thrsh
            print(f'  {str(_ia_p):<30} {_E:>14.10f}  {_dE:>+14.6e}  '
                    f'{" degenerate with ref" if _close else ""}')

    list_selected_sgen_groups = []
    list_selected_basis_groups = []
    list_ref_degenerate_groups = []

    _seen_ia_keys_dedup = set()
    _ia_ex_dedup = []
    for (ia_item, ex_csf_item, E_item) in excitation_candidates:
        if ia_item is None:
            _ia_ex_dedup.append((ia_item, ex_csf_item, E_item))
            continue
        _key_dd = tuple(tuple(p) if isinstance(p, list) else (p,)
                        for p in ia_item)
        if _key_dd not in _seen_ia_keys_dedup:
            _seen_ia_keys_dedup.add(_key_dd)
            _ia_ex_dedup.append((ia_item, ex_csf_item, E_item))
        else:
            print(f'  [ia_ex dedup] dropped duplicate ia_pair {ia_item} '
                  f'(already seen from another ref CSF in degenerate group)')
    excitation_candidates = _ia_ex_dedup

    visited = set()
    ia_groups = []
    for i, (ia_pair_i, ex_csf_i, E_i) in enumerate(excitation_candidates):
        if i in visited:
            continue
        if ia_pair_i is None:
            group = ['None']
        else:
            group = [ia_pair_i]
        visited.add(i)
        for j, (ia_pair_j, ex_csf_j, E_j) in enumerate(excitation_candidates):
            if j not in visited and abs(E_i - E_j) < degen_thrsh:
                group.append(ia_pair_j)
                visited.add(j)
        if 'None' in group and len(group) <= 1:
            continue
        ia_groups.append(group)
        print(f'Degenerate group (E_ex={E_i:.10f}): {group}')

    for symmetry_group in ia_groups:
        group_in_active_space = True
        skip_group = False
        for ia_pair in symmetry_group:
            if ia_pair == 'None':
                continue
            for pair in ia_pair:
                if pair[0] in list_mo_exclud or pair[1] in list_mo_exclud:
                    print(f'{ia_pair} contains mo in list_mo_exclud. Skipping group.')
                    skip_group = True
                    break
                if pair[0] < actmo_start or pair[1] > actmo_end:
                    group_in_active_space = False
            if skip_group: break
        if skip_group: continue

        _ref_csf_list_sel = [list_CSF[ci] for ci in csf_to_ref_grp.get(iCSF, [iCSF])]
        improve, isign_list, isign_ref_by_iapair = _compute_improve_for_group(symmetry_group, CSF, nCSF, Hmat_CSF, Enuc, obt, tbt,
            list_CSF, eigen, evals_CSF, iCSF, nparal,ref_csf_list=_ref_csf_list_sel)
        if improve is None:
            print(f'Group {symmetry_group} produces no excited CSFs. Skipping.')
            continue
        if improve > 1.0e-8:
            print(f'Strange! E0 is larger with group by {improve}. Bombing out!')
            sys.exit()
        group_isign = isign_list[1] if len(isign_list) >= 2 else +1
        print(f'Group mean improve: {improve:.8f}, isign={group_isign}')
        for ia_pair in symmetry_group:
            if ia_pair == 'None':
                continue
            list_improvement[iCSF].append(improve)
            list_improvement_ia[iCSF].append(ia_pair)

        if abs(improve) >= Ethrsh or (l_include_ia_in_CAS and group_in_active_space):
            goes_to_generators = abs(improve) < ratio * Ethrsh
            ref_is_in_group = ('None' in symmetry_group)
            excitation_pairs = [p for p in symmetry_group if p != 'None']
            if ref_is_in_group:
                for ia_pair in excitation_pairs:
                    print(f'{ia_pair} folded into symmetric reference (degenerate with ref)')
                    list_improve.append(improve)
                    list_ia_one_CSF.append(ia_pair)
                list_ref_degenerate_groups.append((excitation_pairs, isign_ref_by_iapair))
            else:
                for ia_pair in excitation_pairs:
                    if goes_to_generators:
                        print(f'{ia_pair} included in S_gen (group)')
                        list_improve.append(improve)
                        list_ia_one_CSF.append(ia_pair)
                    else:
                        print(f'{ia_pair} included in S_extend (group)')
                        list_basis_improve.append(improve)
                if goes_to_generators and len(excitation_pairs) > 1:
                    list_selected_sgen_groups.append((excitation_pairs, group_isign))
                elif not goes_to_generators:
                    for ia_pair in excitation_pairs:
                        list_ia_basis_one_CSF.append(ia_pair)
                    if len(excitation_pairs) > 1:
                        list_selected_basis_groups.append((excitation_pairs, group_isign))
            if l_include_ia_in_CAS and group_in_active_space:
                print(f'Group included because in active space')
        else:
            print(f'Group excluded (|improve|={abs(improve):.2e} < threshold {Ethrsh:.2e})')

    print(f'\n--- Degenerate group isign summary for CSF{iCSF} ---')
    for degen_group_info in list_selected_sgen_groups:
        grp, sgn = degen_group_info
        print(f'  S_gen  isign={sgn:+d}  group={grp}')

    if list_selected_basis_groups:
        _basis_key_to_improve = {ia_hashable_key(ia): imp
                                 for ia, imp in zip(list_ia_basis_one_CSF, list_basis_improve)}
        _used_basis_keys = set()
        _new_basis_list = []
        _new_basis_improve = []
        for group, isign_g in list_selected_basis_groups:
            if len(group) < 2:
                continue
            _k0 = ia_hashable_key(group[0])
            _k1 = ia_hashable_key(group[1])
            if _k0 in _basis_key_to_improve and _k1 in _basis_key_to_improve:
                eff = [(1.0 / np.sqrt(2), group[0]),
                       (float(isign_g) / np.sqrt(2), group[1])]
                _new_basis_list.append(eff)
                _new_basis_improve.append(
                    min(_basis_key_to_improve[_k0], _basis_key_to_improve[_k1]))
                _used_basis_keys.update([_k0, _k1])
                print(f'  [basis-ext] merged degenerate pair into effective pair: '
                      f'({group[0]}+{isign_g}·{group[1]})/√2')
        for ia, imp in zip(list_ia_basis_one_CSF, list_basis_improve):
            if ia_hashable_key(ia) not in _used_basis_keys:
                _new_basis_list.append(ia)
                _new_basis_improve.append(imp)
        list_ia_basis_one_CSF = _new_basis_list
        list_basis_improve = _new_basis_improve
        print(f'  [basis-ext] after merging: {list_ia_basis_one_CSF}')

    return (list_ia_one_CSF, list_improve,list_ia_basis_one_CSF, list_basis_improve,list_selected_sgen_groups, list_ref_degenerate_groups)

def select_ia_pairs_for_CSF_Eavg_with_sym_CSF(list_CSF, list_SOMO_DMO, grouped_ia_pairs, Enuc, obt, tbt, l_include_ia_in_CAS, actmo_start, actmo_end, ratio, eigen, degenerate_partners, combo_order, l_axial_sym=False,list_sym_sign=[], Ethrsh=1.0e-4, list_mo_exclud=[], nparal=1, mo_ml=None, dinfh_degenerate_groups=None, debug=False):
    """
    Select ia pairs for each CSF for their individual lowerings of the uniform weighted lowest eigen eigenvalues of the CSF space.
    The CSFs may be paired with their symmetry partners
    """

    if debug:
        print('\nIn select_ia_pairs_for_CSF_Eavg_with_sym_CSF')

    if nparal > 1:
        Hmat_CSF = construct_Hmat_CSFs_paral_triu(list_CSF, Enuc, obt, tbt, nparal)
    else:
        Hmat_CSF = construct_Hmat_CSFs(list_CSF, Enuc, obt, tbt)

    evals_CSF = get_lowest_n_eigen(Hmat_CSF, eigen)
    nCSF = len(list_CSF)
    (list_list_genmat, list_list_ia_pair, list_list_ex_space,list_list_symadapted_ex_space, list_sym_CSF_vec,
     list_list_extra_vecs, list_is_basis_ext_group, list_list_frozen_positions) = ([] for _ in range(8))
    list_improvement = [[] for _ in range(nCSF)]
    list_improvement_ia = [[] for _ in range(nCSF)]
    n_sym_CSF = 0

    csf_to_ref_grp = build_csf_to_ref_group(Hmat_CSF, nCSF, mo_ml)
    processed_ref_csfs = set()

    for iCSF in range(nCSF):
        if mo_ml is not None and iCSF in processed_ref_csfs:
            continue
        list_ia_one_CSF = []
        list_ia_basis_one_CSF = []
        list_improve = []
        list_basis_improve = []
        n_sym_CSF += 1
        print(f'\nSelecting ia pairs for CSF{iCSF}')
        CSF = list_CSF[iCSF]
        _t_csf0 = time.time()

        degen_thrsh = 0 if mo_ml is None else 1e-8

        _t_scan0 = time.time()
        if mo_ml is not None:
            (list_ia_one_CSF, list_improve,list_ia_basis_one_CSF, list_basis_improve,
             list_selected_sgen_groups, list_ref_degenerate_groups) = \
                select_ia_pairs_sym_aware(CSF, iCSF, grouped_ia_pairs,Enuc, obt, tbt, nparal,
                    l_include_ia_in_CAS, actmo_start, actmo_end, ratio, Ethrsh, list_mo_exclud,
                    nCSF, Hmat_CSF, list_CSF, eigen, evals_CSF,csf_to_ref_grp, list_improvement, list_improvement_ia)
        else:
            list_selected_sgen_groups = []
            list_ref_degenerate_groups = []
            handled_as_pair = set()

            # Phase 1: cheap sequential filter. No cross-pair state (handled_as_pair
            # is never populated), so filtering order doesn't affect the result.
            _valid_pair_candidates = []
            for ia_pair in grouped_ia_pairs:
                pair_in_active_space = True
                pair_is_excluded = False
                for pair in ia_pair:
                    if pair[0] < actmo_start or pair[1] > actmo_end:
                        pair_in_active_space = False
                    if pair[0] in list_mo_exclud or pair[1] in list_mo_exclud:
                        pair_is_excluded = True
                if pair_is_excluded:
                    print(f'{ia_pair} contain mo in list_mo_exclud: {list_mo_exclud}. Next.')
                    continue

                flat_ia = [p for item in ia_pair for p in (item if isinstance(item[0], list) else [item])]
                a_orbs = [p[1] for p in flat_ia]
                if any(a in handled_as_pair for a in a_orbs):
                    print(f'{ia_pair} orbital already handled as degenerate pair, skipping')
                    continue
                _valid_pair_candidates.append((ia_pair, pair_in_active_space))

            # Phase 2: each pair's improve is independent of every other pair's —
            # dispatch across all pairs at once. nparal=1 internally to avoid
            # nested-parallelism oversubscription (the outer Parallel already
            # uses all nparal workers).
            print(f'[PARALLEL] CSF{iCSF}: dispatching {len(_valid_pair_candidates)} ia_pair evals across {nparal} workers')
            if nparal > 1 and len(_valid_pair_candidates) > 0:
                _pair_improves = Parallel(n_jobs=nparal, prefer='threads')(
                    delayed(_compute_improve_for_pair)(CSF, ia_pair, 1, nCSF, Hmat_CSF, Enuc, obt, tbt,
                        list_CSF, eigen, evals_CSF)
                    for (ia_pair, _pias) in _valid_pair_candidates)
            else:
                _pair_improves = [_compute_improve_for_pair(CSF, ia_pair, 1, nCSF, Hmat_CSF, Enuc, obt, tbt,
                        list_CSF, eigen, evals_CSF)
                    for (ia_pair, _pias) in _valid_pair_candidates]

            # Phase 3: sequential decide, exactly the original logic/order/prints.
            for (ia_pair, pair_in_active_space), improve in zip(_valid_pair_candidates, _pair_improves):
                if improve is None:
                    print(f'{ia_pair} does not create any excited CSFs. Now jumping to the next pair.')
                    continue
                if improve > 1.0e-8:
                    print(f'Strange! E0 is larger with the extra CSFs by {improve}. Bombing out!')
                    sys.exit()
                print(f'{ia_pair} Avg energy improvement: {improve:.8f}')
                if not (abs(improve) >= Ethrsh) and not (l_include_ia_in_CAS and pair_in_active_space):
                    print(f'{ia_pair} excluded')
                if abs(improve) >= Ethrsh or (l_include_ia_in_CAS and pair_in_active_space):
                    print(f'{ia_pair} included')
                    if abs(improve) < ratio * Ethrsh:
                        list_improve.append(improve)
                        list_ia_one_CSF.append(ia_pair)
                    else:
                        list_basis_improve.append(improve)
                        list_ia_basis_one_CSF.append(ia_pair)
                    if l_include_ia_in_CAS and pair_in_active_space:
                        print(f'{ia_pair} included because being in active space')
                list_improvement[iCSF].append(improve)
                list_improvement_ia[iCSF].append(ia_pair)

        _t_scan1 = time.time()
        print(f'[TIMING] CSF{iCSF}: pair-scan loop took {_t_scan1 - _t_scan0:.3f}s '
              f'({len(grouped_ia_pairs)} candidate pairs)')

        _t_combo0 = time.time()
        _n_combos_evaluated = 0
        if len(list_ia_basis_one_CSF) >= 1:
            from itertools import combinations as icombs
            print(f'\nEvaluating chained combinations of {len(list_ia_basis_one_CSF)} '
                  f'basis-extension pairs for CSF{iCSF}')
            print(f'  Basis-extension pairs: {list_ia_basis_one_CSF}')

            n_basis = len(list_ia_basis_one_CSF)

            list_combo_improve = []
            list_combo_pairs  = []
            seen_index_sets = set()

            # Phase 1: enumerate every (r, combo) up front. list_ia_basis_one_CSF
            # only ever grows with indices >= n_basis during this block, and
            # icombs(range(n_basis), r) is fixed to the original n_basis, so no
            # combo generated here ever depends on another combo's outcome.
            _combo_jobs = []
            for r in range(2, min(combo_order + 1, n_basis + 1)):
                for combo in icombs(range(n_basis), r):
                    combo_pairs = [list_ia_basis_one_CSF[i] for i in combo]
                    _combo_jobs.append((r, combo_pairs))
            _n_combos_evaluated = len(_combo_jobs)

            def _evaluate_one_combo(r, combo_pairs):
                """Independent per-combo work: build the trial state, then its
                Hamiltonian row/diagonalization. Returns
                (combo_has_eff, key, improve_combo) with improve_combo=None if
                the combo annihilates/cancels. No shared state is touched here —
                safe to run in any order or in parallel."""
                combo_has_eff = any(is_effective_pair(p) for p in combo_pairs)
                key = None
                flat_pairs = []
                if not combo_has_eff:
                    for item in combo_pairs:
                        if isinstance(item[0], list):
                            for pair in item:
                                flat_pairs.append(pair)
                        else:
                            flat_pairs.append(item)
                    all_i = [item[0] for item in flat_pairs]
                    all_a = [item[1] for item in flat_pairs]
                    key = (frozenset(all_i), frozenset(all_a))

                if combo_has_eff:
                    import copy as _copy
                    sds_chain = _copy.deepcopy(list(CSF[0]))
                    coefs_chain = np.array(CSF[2], dtype=float)
                    valid = True
                    for eff_item in combo_pairs:
                        sds_chain, coefs_chain = apply_effective_pair(sds_chain, coefs_chain, eff_item)
                        if sds_chain is None:
                            valid = False
                            break
                    if valid:
                        sds_chain, idx_chain, coefs_chain = canonicalize_chain_state(sds_chain, coefs_chain)
                        if sds_chain is None:
                            valid = False
                    if not valid:
                        return (combo_has_eff, key, None)
                    T_CSF_current = [sds_chain, idx_chain, coefs_chain]
                else:
                    T_CSF_current = CSF
                    valid = True
                    stored_pair_0, stored_pair_1 = [], []
                    for pair in flat_pairs:
                        i, a = pair[0], pair[1]
                        T_op = get_Tiiaa_00(i, a)
                        T_CSF_current = op_action_tz_CSF(T_op, T_CSF_current)
                        norm_current = np.sqrt(overlap_CSFs(T_CSF_current, T_CSF_current))
                        if norm_current < 1e-9 or pair[0] in stored_pair_0 or pair[1] in stored_pair_1:
                            valid = False
                            break
                        stored_pair_0.append(pair[0])
                        stored_pair_1.append(pair[1])
                    if not valid:
                        return (combo_has_eff, key, None)

                norm_final = np.sqrt(overlap_CSFs(T_CSF_current, T_CSF_current))
                onlist_T, idxlist_T, coefvec_T = T_CSF_current
                T_CSF_normalized = [onlist_T, idxlist_T, coefvec_T / norm_final]

                ndim_combo = nCSF + 1
                Hmat_combo = np.zeros([ndim_combo, ndim_combo])
                Hmat_combo[0:nCSF, 0:nCSF] = Hmat_CSF
                Hmat_combo[nCSF:, nCSF:] = construct_Hmat_CSFs([T_CSF_normalized], Enuc, obt, tbt)
                for jCSF, CSForig in enumerate(list_CSF):
                    Helm = Helm_between_CSFs(Enuc, obt, tbt, CSForig, T_CSF_normalized)
                    Hmat_combo[nCSF, jCSF] = Helm
                    Hmat_combo[jCSF, nCSF] = Helm

                evals_combo = get_lowest_n_eigen(Hmat_combo, eigen)
                improve_combo = (evals_combo - evals_CSF).mean()
                return (combo_has_eff, key, improve_combo)

            # Phase 2: dispatch every combo's independent work across all
            # workers at once (kept serial internally per combo to avoid nested
            # parallelism / oversubscription against the outer Parallel).
            print(f'[PARALLEL] CSF{iCSF}: dispatching {_n_combos_evaluated} combo evals across {nparal} workers')
            if nparal > 1 and _n_combos_evaluated > 0:
                _combo_results = Parallel(n_jobs=nparal, prefer='threads')(
                    delayed(_evaluate_one_combo)(r, combo_pairs) for (r, combo_pairs) in _combo_jobs)
            else:
                _combo_results = [_evaluate_one_combo(r, combo_pairs) for (r, combo_pairs) in _combo_jobs]

            # Phase 3: sequential decide — replicates the exact original
            # order-dependent logic (dedup check first, then validity, then
            # threshold), using the precomputed results.
            for (r, combo_pairs), (combo_has_eff, key, improve_combo) in zip(_combo_jobs, _combo_results):
                print('Combination: ', combo_pairs)

                if key is not None and key in seen_index_sets:
                    print(key)
                    print(f'  Skipping {combo_pairs}: same creation/annihilation '
                          f'index sets as previous combo')
                    continue

                if improve_combo is None:
                    print(f'  Combo {combo_pairs} annihilated/cancelled, skipping')
                    continue

                if improve_combo > 1.0e-8:
                    print(f' Strange! Combo {combo_pairs} raised energy by {improve_combo}. Skipping.')
                    continue

                list_combo_improve.append(improve_combo)
                list_combo_pairs.append(combo_pairs)
                print(f' Combo {combo_pairs}: avg subspace energy change = {improve_combo:.8f}')

                epsilon_ell = (ratio*Ethrsh)**r
                if abs(improve_combo) >= epsilon_ell:
                    if combo_has_eff:
                        print(f'pairs — skipping add to basis list (handled by _icombs loop)')
                        continue
                    print(f' Combo {combo_pairs} surpasses threshold ({epsilon_ell:.2e}), '
                          f'adding chained CSF to basis extension list')
                    list_ia_basis_one_CSF.append(combo_pairs)
                    list_basis_improve.append(improve_combo)
                    if key is not None: seen_index_sets.add(key)
                    # NOTE: the mo_ml degenerate-partner-chain logic from the
                    # original code is intentionally omitted here — this whole
                    # block only executes when mo_ml is None (the sym-aware
                    # path is handled separately by select_ia_pairs_sym_aware),
                    # so that branch was always dead code on this path.
                else:
                    print(f' Combo {combo_pairs} energy {abs(improve_combo):.2e} below threshold ({epsilon_ell:.2e}), skipping')

            print(f'\n--- Chained combination summary for CSF{iCSF} ---')
            print(f'  Single basis-extension pairs:')
            for bp, bi in zip(list_ia_basis_one_CSF, list_basis_improve):
                print(f'    {bp}  improve={bi:.8f}')
            print(f'  Chained combinations:')
            if len(list_combo_pairs) == 0:
                print(f'    (none valid)')
            for cp, ci in zip(list_combo_pairs, list_combo_improve):
                delta = ci - sum(
                    list_basis_improve[list_ia_basis_one_CSF.index(p)]
                    for p in cp if p in list_ia_basis_one_CSF
                )
                print(f'    {cp}  improve={ci:.8f}  '
                      f'vs sum-of-singles={sum(list_basis_improve[list_ia_basis_one_CSF.index(p)] for p in cp if p in list_ia_basis_one_CSF):.8f}  '
                      f'delta={delta:.8f}')

        _t_combo1 = time.time()
        print(f'[TIMING] CSF{iCSF}: chain-combination loop took {_t_combo1 - _t_combo0:.3f}s '
              f'({_n_combos_evaluated} combos evaluated)')
        _t_post0 = time.time()

        if len(list_ia_one_CSF) != 0:
            print(f'Before sorting — generator ia_pairs and improve for CSF {iCSF}:')
            for ia_p, imp in zip(list_ia_one_CSF, list_improve):
                print(f'  {ia_p}  improve={imp:.8f}  |improve|={abs(imp):.8f}')
            sorted_list_improve, sorted_list_ia_one_CSF = zip(*sorted(zip(list_improve, list_ia_one_CSF), key=lambda x: x[0]))
            sorted_list_ia_one_CSF = list(sorted_list_ia_one_CSF)
            print(f'After sorting (ascending improve = largest |improve| first) — generator ia_pairs included for CSF {iCSF}:')
            for ii, ia_pairs in enumerate(sorted_list_ia_one_CSF):
                print(f'  pos {ii}: {ia_pairs}  improve={sorted_list_improve[ii]:.8f}')
        else:
            sorted_list_ia_one_CSF = []

        if len(list_ia_basis_one_CSF) != 0:
            sorted_list_basis_improve, sorted_list_ia_basis_one_CSF = zip(
                *sorted(zip(list_basis_improve, list_ia_basis_one_CSF), key=lambda x: x[0]))
            sorted_list_ia_basis_one_CSF = list(sorted_list_ia_basis_one_CSF)
            print(f'ia_pairs for basis extension included for CSF {iCSF}:')
            for ii, ia_pairs in enumerate(sorted_list_ia_basis_one_CSF):
                print(ia_pairs, sorted_list_basis_improve[ii])
        else:
            sorted_list_ia_basis_one_CSF = []

        list_zero_theta = [0.0] * len(sorted_list_ia_one_CSF)
        if mo_ml is not None and len(csf_to_ref_grp.get(iCSF, [iCSF])) > 1:
            ref_grp = csf_to_ref_grp[iCSF]
            processed_ref_csfs.update(ref_grp)
            N_ref = len(ref_grp)
            print(f'\nBuilding combined ex_space for degenerate reference group 'f'{ref_grp} (N={N_ref})')

            H_csr = csr_matrix(Hmat_CSF) if not scipy.sparse.issparse(Hmat_CSF) else Hmat_CSF
            _, _psi_gs_vec = scipy.sparse.linalg.eigsh(H_csr, k=1, which='SA')
            psi_gs_full = _psi_gs_vec[:, 0]
            print(f'  Full CSF-space GS eigenvector: {psi_gs_full}')

            sym_vec_ref_raw = np.array([psi_gs_full[ci] for ci in ref_grp])
            nrm_raw = np.linalg.norm(sym_vec_ref_raw)
            if nrm_raw > 1e-12:
                signs_raw = np.sign(sym_vec_ref_raw)
                if np.all(signs_raw == signs_raw[0]):
                    sym_vec_ref = signs_raw[0] * np.ones(N_ref) / np.sqrt(N_ref)
                    print(f'  GS projection onto ref_grp {ref_grp}: raw={sym_vec_ref_raw}; '
                          f'all-same-sign → enforcing equal weights: {sym_vec_ref}')
                else:
                    sym_vec_ref = sym_vec_ref_raw / nrm_raw
                    print(f'  GS projection onto ref_grp {ref_grp}: mixed signs, '
                          f'using raw projection: {sym_vec_ref}')
            else:
                sym_vec_ref = np.ones(N_ref) / np.sqrt(N_ref)
                print(f'  WARNING: GS projection onto ref_grp {ref_grp} is zero; '
                      f'using equal superposition: {sym_vec_ref}')

            if 'list_selected_sgen_groups' in dir() and len(list_selected_sgen_groups) > 0:
                used_keys_dg = set()
                effective_ia_pairs_dg = []
                for group, isign_g in list_selected_sgen_groups:
                    if len(group) < 2:
                        continue
                    # Use pre-computed isign from the augmented-H eigenvector (reliable,
                    # and avoids the need for a multi-occupancy reference state).
                    effective_ia_pairs_dg.append(
                        [(1.0/np.sqrt(2), group[0]),
                         (float(isign_g)/np.sqrt(2), group[1])])
                    for g in [group[0], group[1]]:
                        k_key = (tuple(g) if not isinstance(g[0], list)
                                 else tuple(tuple(p) for p in g))
                        used_keys_dg.add(k_key)
                    print(f'  [degen-ref] effective symad pair: '
                          f'({group[0]}+{isign_g}·{group[1]})/√2')
                for ia in sorted_list_ia_one_CSF:
                    k_key = (tuple(ia) if not isinstance(ia[0], list)
                             else tuple(tuple(p) for p in ia))
                    if k_key not in used_keys_dg:
                        effective_ia_pairs_dg.append(ia)
            else:
                effective_ia_pairs_dg = list(sorted_list_ia_one_CSF)
            print(f'\n=== Degen-ref {ref_grp}: {len(effective_ia_pairs_dg)} effective ia-pairs ===')
            for _ep_i, _ep in enumerate(effective_ia_pairs_dg):
                if len(_ep) > 0 and isinstance(_ep[0], tuple):
                    print(f'  [{_ep_i}] SYM-ADAPTED: {[(round(float(_c),4), _ia) for _c,_ia in _ep]}')
                else:
                    print(f'  [{_ep_i}] plain: {_ep}')

            list_ex_spaces_per_csf = []
            list_genmats_per_csf   = []
            for ci in ref_grp:
                CSFi = list_CSF[ci]
                ex_sp_i, gm_i = make_pair_ex_space_manual(
                    CSFi, effective_ia_pairs_dg, [], nparal, build_genmat=True)
                list_ex_spaces_per_csf.append(ex_sp_i)
                list_genmats_per_csf.append(gm_i)

            # ── 4. Merge into combined ex_space (O(1) dedup by SD-index key) ──────
            combined_ex_space = [list_CSF[ci] for ci in ref_grp]
            _csf_key = lambda csf: frozenset(int(x) for x in csf[1])
            combined_key_to_idx = {_csf_key(csf): idx
                                   for idx, csf in enumerate(combined_ex_space)}
            local_to_combined = []
            for k, ci in enumerate(ref_grp):
                mapping = [k]
                for ex_csf in list_ex_spaces_per_csf[k][1:]:
                    ex_key = _csf_key(ex_csf)
                    if ex_key in combined_key_to_idx:
                        found = combined_key_to_idx[ex_key]
                    else:
                        found = len(combined_ex_space)
                        combined_ex_space.append(ex_csf)
                        combined_key_to_idx[ex_key] = found
                    mapping.append(found)
                local_to_combined.append(mapping)
            ndim_combined = len(combined_ex_space)
            print(f'  Combined ex_space dim: {ndim_combined} '
                  f'({N_ref} refs + {ndim_combined - N_ref} unique excitations)')

            sym_CSF_vec = np.zeros(ndim_combined)
            for k in range(N_ref):
                sym_CSF_vec[k] = sym_vec_ref[k]
            nrm = np.linalg.norm(sym_CSF_vec)
            if nrm > 1e-12:
                sym_CSF_vec /= nrm

            H_restricted = Hmat_CSF[np.ix_(ref_grp, ref_grp)]
            _evals_r, evecs_r = np.linalg.eigh(H_restricted)
            print(f'  Restricted H eigenvalues (for symad_states_g): {_evals_r}')
            _ov_ev = [abs(float(np.dot(evecs_r[:, i], sym_vec_ref))) for i in range(N_ref)]
            ev_order = sorted(range(N_ref), key=lambda i: -_ov_ev[i])
            print(f'  Restricted H eigenvector overlaps with sym_vec_ref: '
                  f'{[f"{_ov_ev[i]:.4f}" for i in range(N_ref)]}; '
                  f'ordering for symad_states_g: {ev_order}')
            max_local = max(len(m) for m in local_to_combined)
            symad_states_g = []
            for ev_idx in ev_order:            # most-aligned with sym_vec_ref first
                sym_vec_ev_raw = evecs_r[:, ev_idx]
                signs_ev = np.sign(sym_vec_ev_raw)
                sym_vec_ev = signs_ev / np.sqrt(N_ref)   # exact ±1/√N for every component
                print(f'  [symad_states_g] ev_idx={ev_idx}: raw={sym_vec_ev_raw}, '
                      f'signs={signs_ev.tolist()}, enforced={sym_vec_ev.tolist()}')
                ref_vec_ev = np.zeros(ndim_combined)
                for k in range(N_ref):
                    ref_vec_ev[k] = sym_vec_ev[k]
                nrm_ev = np.linalg.norm(ref_vec_ev)
                if nrm_ev < 1e-12:
                    continue
                ref_vec_ev /= nrm_ev
                symad_states_g.append(make_UCSF_state(combined_ex_space, ref_vec_ev))
                for p in range(1, max_local):
                    vec_p = np.zeros(ndim_combined)
                    for k in range(N_ref):
                        if p < len(local_to_combined[k]):
                            vec_p[local_to_combined[k][p]] += sym_vec_ev[k]
                    nrm_p = np.linalg.norm(vec_p)
                    if nrm_p > 1e-12:
                        symad_states_g.append(make_UCSF_state(combined_ex_space, vec_p / nrm_p))
            print(f'Symadapted states ({len(symad_states_g)}) for degenerate reference group {ref_grp} (N={N_ref})')

            n_ia = len(effective_ia_pairs_dg)
            combined_genmats = []
            for p in range(n_ia):
                G_comb = csr_matrix((ndim_combined, ndim_combined), dtype=float)
                for k in range(N_ref):
                    gm_k = list_genmats_per_csf[k]
                    if p < len(gm_k):
                        G_mapped = remap_genmat_to_combined(gm_k[p], local_to_combined[k], ndim_combined)
                    else:
                        G_mapped = csr_matrix((ndim_combined, ndim_combined), dtype=float)
                    G_comb = G_comb + G_mapped
                assert np.isclose(scipy.sparse.linalg.norm(G_comb + G_comb.transpose()), 0.0), \
                    f'Combined genmat {p} for ref_grp {ref_grp} is not antisymmetric'
                combined_genmats.append(G_comb)

            print(f'  [degen-ref] {n_ia} effective ia-pairs, {len(combined_genmats)} combined genmats')
            assert n_ia == len(combined_genmats), \
                f'ia_pair/genmat count mismatch: {n_ia} vs {len(combined_genmats)}'

            list_list_ex_space.append(combined_ex_space)
            list_list_ia_pair.append(effective_ia_pairs_dg)
            list_list_genmat.append(combined_genmats)
            list_sym_CSF_vec.append(sym_CSF_vec)
            print(f' Adding {len(symad_states_g)} symad states to list_list_symadapted_ex_space '
                  f'for degenerate reference group {ref_grp} (N={N_ref})')
            list_list_symadapted_ex_space.append(symad_states_g)
            list_list_extra_vecs.append([])
            list_is_basis_ext_group.append(False)
            list_list_frozen_positions.append([])   # ref combination encoded in sym_CSF_vec
        else:
            CSF = list_CSF[iCSF]

            if mo_ml is not None and 'list_selected_sgen_groups' in dir():
                used_keys = set()
                effective_ia_pairs = []
                for group, isign_g in list_selected_sgen_groups:
                    if len(group) < 2:
                        continue
                    _, _, ex_sp_0, _, _ = prepare_UCSF_for_one_CSF(CSF, [group[0]], [0.0], nparal, False)
                    _, _, ex_sp_1, _, _ = prepare_UCSF_for_one_CSF(CSF, [group[1]], [0.0], nparal, False)
                    correct_isign = isign_g
                    print(f'  ia-pair isign_ext={correct_isign}  '
                          f'ref isign (from ref_degen_groups) would need checking for consistency')
                    if len(ex_sp_0) > 1 and len(ex_sp_1) > 1:
                        G0_state, G1_state = ex_sp_0[1], ex_sp_1[1]
                        for j_ext, csf_ext in enumerate(list_CSF):
                            h0 = Helm_between_CSFs(Enuc, obt, tbt, csf_ext, G0_state)
                            h1 = Helm_between_CSFs(Enuc, obt, tbt, csf_ext, G1_state)
                            if abs(h0 + h1) > 1e-10:
                                correct_isign = +1
                                print(f'  ia-pair symad: isign=+1 (coupling to CSF {j_ext})')
                                break
                            if abs(h0 - h1) > 1e-10:
                                correct_isign = -1
                                print(f'  ia-pair symad: isign=-1 (coupling to CSF {j_ext})')
                                break
                    effective_ia_pairs.append([(1.0/np.sqrt(2), group[0]), (float(correct_isign)/np.sqrt(2), group[1])])
                    k0 = (tuple(group[0]) if not isinstance(group[0][0], list)
                          else tuple(tuple(p) for p in group[0]))
                    k1 = (tuple(group[1]) if not isinstance(group[1][0], list)
                          else tuple(tuple(p) for p in group[1]))
                    used_keys.add(k0)
                    used_keys.add(k1)
                    print(f'  Effective symad pair: ({group[0]}+{correct_isign}·{group[1]})/√2')
                for ia in sorted_list_ia_one_CSF:
                    k = (tuple(ia) if not isinstance(ia[0], list)
                         else tuple(tuple(p) for p in ia))
                    if k not in used_keys:
                        effective_ia_pairs.append(ia)
            else:
                effective_ia_pairs = list(sorted_list_ia_one_CSF)

            print(f'\n=== CSF{iCSF} effective_ia_pairs ({len(effective_ia_pairs)} total) ===')
            for k, ep in enumerate(effective_ia_pairs):
                if len(ep) > 0 and isinstance(ep[0], tuple):
                    print(f'  [{k}] SYM-ADAPTED: {[(round(c,4), ia) for c,ia in ep]}')
                else:
                    print(f'  [{k}] plain:       {ep}')

            _t_exspace0 = time.time()
            list_pair_ex_space, list_genmat = make_pair_ex_space_manual(CSF, effective_ia_pairs, [], nparal, build_genmat=True, debug=debug, combo_order=combo_order)
            _t_exspace1 = time.time()
            print(f'[TIMING] CSF{iCSF}: make_pair_ex_space_manual took {_t_exspace1 - _t_exspace0:.3f}s')

            _tol_gs = 1e-8
            _sd_lookup = {}
            for _st in list_pair_ex_space:
                for _sd_arr, _sd_idx in zip(_st[0], _st[1]):
                    _sd_lookup[int(_sd_idx)] = _sd_arr

            _order = sorted(range(len(list_pair_ex_space)),
                            key=lambda _i: len(list_pair_ex_space[_i][1]))
            _Q_vecs   = []
            _Q_states = []

            for _orig_i in _order:
                _st  = list_pair_ex_space[_orig_i]
                _vec = {int(k): float(c) for k, c in zip(_st[1], _st[2])}

                # project out each basis vector already in Q
                for _q in _Q_vecs:
                    _ov_qi = sum(_q.get(_sd, 0.0) * _vec.get(_sd, 0.0) for _sd in _q)
                    if abs(_ov_qi) > _tol_gs:
                        for _sd, _qc in _q.items():
                            _vec[_sd] = _vec.get(_sd, 0.0) - _ov_qi * _qc

                # drop near-zero entries
                _vec = {_sd: _c for _sd, _c in _vec.items() if abs(_c) > _tol_gs}
                _norm = np.sqrt(sum(_c ** 2 for _c in _vec.values()))

                if _norm < _tol_gs:
                    print(f'  [GS] state[{_orig_i}] ({len(_st[1])} SDs) '
                          f'SD-idx={sorted(int(x) for x in _st[1])} '
                          f'→ residual≈0, dropped')
                    continue

                # normalize
                _vec = {_sd: _c / _norm for _sd, _c in _vec.items()}
                _Q_vecs.append(_vec)

                if abs(_norm - 1.0) > 1e-6:
                    # state was modified — rebuild tuple from residual
                    _new_idx   = sorted(_vec.keys())
                    _new_coefs = np.array([_vec[_k] for _k in _new_idx])
                    _new_sds   = [_sd_lookup[_k] for _k in _new_idx]
                    _new_st    = [_new_sds, _new_idx, _new_coefs]
                    _Q_states.append(_new_st)
                    print(f'  [GS] state[{_orig_i}] ({len(_st[1])} SDs) '
                          f'→ replaced by residual ({len(_new_idx)} SDs) '
                          f'SD-idx={_new_idx}')
                else:
                    _Q_states.append(_st)
                    print(f'  [GS] state[{_orig_i}] ({len(_st[1])} SDs) → kept unchanged')

            _n_before = len(list_pair_ex_space)
            list_pair_ex_space = _Q_states
            # _n_before = len(list_pair_ex_space)
            # _Q_states = list_pair_ex_space

            print(f'  [GS] CSF{iCSF}: {_n_before} → {len(list_pair_ex_space)} states')

            _is_sym_p = lambda _p: len(_p) > 0 and isinstance(_p[0], tuple)
            _sub_pairs_gs = []
            _sub_map_gs   = []
            for _eff_idx, _p in enumerate(effective_ia_pairs):
                if _is_sym_p(_p):
                    for (_c_k, _ia_k) in _p:
                        _sub_pairs_gs.append(_ia_k)
                        _sub_map_gs.append((_eff_idx, float(_c_k)))
                else:
                    _sub_pairs_gs.append(_p)
                    _sub_map_gs.append((_eff_idx, 1.0))

            _t_gsgenmat0 = time.time()
            _sub_genmats_gs = make_iapair_genmat_new(_sub_pairs_gs, list_pair_ex_space, nparal)
            _t_gsgenmat1 = time.time()
            print(f'[TIMING] CSF{iCSF}: GS make_iapair_genmat_new took {_t_gsgenmat1 - _t_gsgenmat0:.3f}s')

            _n_eff_gs    = len(effective_ia_pairs)
            _n_basis_gs  = len(list_pair_ex_space)
            _combined_gs = [None] * _n_eff_gs
            for _k, (_eff_idx, _coef) in enumerate(_sub_map_gs):
                _g = _sub_genmats_gs[_k] * _coef
                _combined_gs[_eff_idx] = (_g if _combined_gs[_eff_idx] is None
                                          else _combined_gs[_eff_idx] + _g)
            list_genmat = [_g if _g is not None else csr_matrix((_n_basis_gs, _n_basis_gs))for _g in _combined_gs]
            print(f'  [GS] rebuilt list_genmat: '
                  f'{_n_eff_gs} generators × {_n_basis_gs}×{_n_basis_gs}')

            symad_ex_space = list(list_pair_ex_space)
            list_list_ex_space.append(list_pair_ex_space)
            list_list_symadapted_ex_space.append(symad_ex_space)
            list_list_ia_pair.append(effective_ia_pairs)
            list_list_genmat.append(list_genmat)
            ndim_space = len(list_pair_ex_space)
            sym_CSF_vec = np.zeros(ndim_space)
            sym_CSF_vec[0] = 1.0
            list_sym_CSF_vec.append(sym_CSF_vec)
            list_list_extra_vecs.append([])
            list_is_basis_ext_group.append(False)
            parent_group_global_idx = len(list_list_ex_space) - 1

            _rd_keys = set()
            if 'list_ref_degenerate_groups' in dir():
                for (_rd_pairs, _) in list_ref_degenerate_groups:
                    for _rd_ia in _rd_pairs:
                        _rd_key = tuple(tuple(p) if isinstance(p, list) else p
                                        for p in _rd_ia)
                        _rd_keys.add(_rd_key)
            _is_sym_ep = lambda _ep: len(_ep) > 0 and isinstance(_ep[0], tuple)
            _frozen_pos = []
            for _ep_idx, _ep in enumerate(effective_ia_pairs):
                if _is_sym_ep(_ep):
                    for (_, _sub_ia) in _ep:
                        _sub_key = tuple(tuple(p) if isinstance(p, list) else p
                                         for p in _sub_ia)
                        if _sub_key in _rd_keys:
                            _frozen_pos.append(_ep_idx)
                            break
                else:
                    _ep_key = tuple(tuple(p) if isinstance(p, list) else p
                                    for p in _ep)
                    if _ep_key in _rd_keys:
                        _frozen_pos.append(_ep_idx)
            if _frozen_pos:
                print(f'  [frozen-theta] CSF{iCSF}: positions {_frozen_pos} '
                      f'in effective_ia_pairs fixed at π/4 (ref-degenerate)')
            list_list_frozen_positions.append(_frozen_pos)

            if len(sorted_list_ia_basis_one_CSF) >= 1:
                print(f'Adding chained CSFs for {len(sorted_list_ia_basis_one_CSF)} '
                      f'basis-extension pairs: {sorted_list_ia_basis_one_CSF}')

                chain_data = []  # list of (flat_pairs, T_CSF_normalized, E_chain)

                from itertools import combinations as _icombs

                def _yield_flat_pairs(item):
                    """Recursively yield flat [i, a] pairs from any ia-pair format."""
                    if is_effective_pair(item):
                        for _, _grp in item:
                            for _pp in _grp:
                                yield [int(_pp[0]), int(_pp[1])]
                    elif (isinstance(item, (list, tuple)) and len(item) == 2
                          and not isinstance(item[0], (list, tuple))):
                        yield [int(item[0]), int(item[1])]
                    elif isinstance(item, (list, tuple)):
                        for _sub in item:
                            yield from _yield_flat_pairs(_sub)

                _indiv_pairs = []
                _seen_ip = set()
                for _ibg in sorted_list_ia_basis_one_CSF:
                    for _fp in _yield_flat_pairs(_ibg):
                        _k = (_fp[0], _fp[1])
                        if _k not in _seen_ip:
                            _seen_ip.add(_k)
                            _indiv_pairs.append(_fp)

                _seen_chain_keys = set()
                print(f'{_indiv_pairs} are all the possible excitations for extensions')
                for _r in range(1, min(combo_order + 1, len(_indiv_pairs) + 1)):
                    for _cidx in _icombs(range(len(_indiv_pairs)), _r):
                        _cflat = [_indiv_pairs[_k] for _k in _cidx]
                        _all_i = [_p[0] for _p in _cflat]
                        _all_a = [_p[1] for _p in _cflat]
                        print(f'{_cflat} is being considered for chain combination with order {_r}')
                        if len(set(_all_i)) < len(_all_i) or len(set(_all_a)) < len(_all_a):
                            print(f'Shared orbitals: Skipping ia pair {_cflat}')
                            continue
                        _ckey = (frozenset(_all_i), frozenset(_all_a))
                        if _ckey in _seen_chain_keys:
                            print(f' Repeated chain')
                            continue
                        _Tc = CSF
                        _cvalid = True
                        for _p in _cflat:
                            _Tc = op_action_tz_CSF(get_Tiiaa_00(_p[0], _p[1]), _Tc)
                            if _Tc is None or np.sqrt(overlap_CSFs(_Tc, _Tc)) < 1e-9:
                                print(f' State annihilates')
                                _cvalid = False
                                break
                        if not _cvalid:
                            continue
                        _cn = np.sqrt(overlap_CSFs(_Tc, _Tc))
                        _Tc_norm = [_Tc[0], _Tc[1], _Tc[2] / _cn]
                        _Hmat_c = np.zeros([nCSF + 1, nCSF + 1])
                        _Hmat_c[:nCSF, :nCSF] = Hmat_CSF
                        if nparal > 1:
                            _Hmat_c[nCSF:, nCSF:] = construct_Hmat_CSFs_paral_triu(
                                [_Tc_norm], Enuc, obt, tbt, nparal)
                        else:
                            _Hmat_c[nCSF:, nCSF:] = construct_Hmat_CSFs([_Tc_norm], Enuc, obt, tbt)
                        for _jc, _csorig in enumerate(list_CSF):
                            _h = Helm_between_CSFs(Enuc, obt, tbt, _csorig, _Tc_norm)
                            _Hmat_c[nCSF, _jc] = _h
                            _Hmat_c[_jc, nCSF] = _h
                        _evc = get_lowest_n_eigen(_Hmat_c, eigen)
                        _impc = (_evc - evals_CSF).mean()
                        _eps_ell = ((Ethrsh) ** 3) / 100
                        if abs(_impc) < _eps_ell:
                            print(f'  Combo chain {_cflat}: improve={_impc:.2e} below threshold, skipping')
                            continue
                        _seen_chain_keys.add(_ckey)
                        _Ec = Helm_between_CSFs(Enuc, obt, tbt, _Tc_norm, _Tc_norm)
                        chain_data.append((_cflat, _Tc_norm, _Ec))
                        print(f' Combo chain {_cflat}: improve={_impc:.8f} above threshold, kept')

                # Pass 2: group by equal E_chain, combine degenerate pairs
                chain_visited = set()
                chain_groups = []
                for i, (iag_i, csf_i, E_i) in enumerate(chain_data):
                    if i in chain_visited: continue
                    chain_visited.add(i)
                    group = [(iag_i, csf_i)]
                    for j, (iag_j, csf_j, E_j) in enumerate(chain_data):
                        if j not in chain_visited and abs(E_i - E_j) < degen_thrsh:
                            group.append((iag_j, csf_j))
                            chain_visited.add(j)
                    chain_groups.append(group)

                print(f'\n--- Chain degenerate group summary for CSF{iCSF} ---')
                for cg in chain_groups:
                    if len(cg) == 2:
                        print(f'  Degenerate chain pair: [{cg[0][0]}] <-> [{cg[1][0]}]  (isign computed below)')
                    else:
                        print(f'  Chain: [{cg[0][0]}]')

                _cg_sub_pairs = []
                _cg_sub_map = []   # list of (eff_idx, coef) parallel to _cg_sub_pairs
                for _cg_eff_idx, _cg_p in enumerate(effective_ia_pairs):
                    print(f'Consider the ia pair {_cg_p}')
                    if is_effective_pair(_cg_p):
                        for (_cg_c, _cg_ia) in _cg_p:
                            print(f' The individual ia pairs {_cg_ia}')
                            _cg_sub_pairs.append(_cg_ia[0])
                            _cg_sub_map.append((_cg_eff_idx, float(_cg_c)))
                    else:
                        _cg_sub_pairs.append(_cg_p[0])
                        _cg_sub_map.append((_cg_eff_idx, 1.0))
                _cg_n_eff = len(effective_ia_pairs)

                # All chain excitations from this refCSF share ONE rotation space
                # and ONE genmat (build_combined_chain_excitation_space): the union
                # of every chain determinant and every T_p|chain> as individual SDs.
                # Each chain group is still its OWN basis-extension UCSF, but they
                # all rotate under the SAME unitary U = exp(Σ θ_p G_p), so the chain
                # UCSFs W|chain_a>, W|chain_b> stay orthogonal by construction.  The
                # θ amplitudes are shared with the parent group through
                # shared_unitary_groups (list_is_basis_ext_group → parent_map).
                #
                # NOTE: list_list_symadapted_ex_space (the explicit full-ex H basis)
                # is left per-chain (build_chain_group_excitation_space) — only the
                # rotation space / genmat are combined, per the design rule that the
                # H-basis handling must not touch the rotation space.
                all_chain_csfs = []
                chain_group_member_idx = []   # parallel to chain_groups
                for chain_group in chain_groups:
                    _members = []
                    for _, _ccsf in chain_group:
                        _members.append(len(all_chain_csfs))
                        all_chain_csfs.append(_ccsf)
                    chain_group_member_idx.append(_members)

                shared_ex_sp, shared_gm, chain_vecs = ([], [], [])
                if all_chain_csfs:
                    # Combine the PARENT refCSF rotation space TOGETHER with every
                    # chain excitation space into ONE common SD basis, and build ONE
                    # shared genmat used by the parent AND all chain groups.  With a
                    # single unitary U on the common basis, <W refCSF | W chain> and
                    # <W chain_a | W chain_b> reduce to the bare CSF overlaps (0 for
                    # disjoint SDs) — parent ⊥ chains and chain ⊥ chain by construction.
                    _combined_inputs = [CSF] + all_chain_csfs
                    shared_ex_sp, shared_gm, _all_vecs = build_combined_chain_excitation_space(
                        _combined_inputs, _cg_sub_pairs, _cg_sub_map, _cg_n_eff, nparal)
                    ref_vec    = _all_vecs[0]
                    chain_vecs = _all_vecs[1:]            # parallel to all_chain_csfs
                    print(f'  Combined rotation space (refCSF + chains): {len(shared_ex_sp)} '
                          f'individual SDs, one shared genmat across parent + '
                          f'{len(chain_groups)} chain group(s)')

                    # Re-point the already-appended parent group onto the shared
                    # rotation space / genmat (its symadapted H-basis is untouched).
                    list_list_ex_space[parent_group_global_idx] = shared_ex_sp
                    list_list_genmat[parent_group_global_idx]   = shared_gm
                    list_sym_CSF_vec[parent_group_global_idx]   = ref_vec
                    print(f'    Re-pointed parent group {parent_group_global_idx} onto '
                          f'shared rotation space ({len(shared_ex_sp)} SDs)')

                for _gidx, chain_group in enumerate(chain_groups):
                    _members = chain_group_member_idx[_gidx]
                    if mo_ml is not None and len(chain_group) == 2:
                        # Degenerate chain pair: variational symmetric/antisymmetric
                        # combination of the two chain reference vectors.
                        _, csf_a = chain_group[0]
                        _, csf_b = chain_group[1]

                        # Determine isign from augmented H eigenvector
                        Hmat_chain_aug = np.zeros([nCSF + 2, nCSF + 2])
                        Hmat_chain_aug[0:nCSF, 0:nCSF] = Hmat_CSF
                        Hmat_chain_aug[nCSF:, nCSF:] = construct_Hmat_CSFs([csf_a, csf_b], Enuc, obt, tbt)
                        for k, csf_ex in enumerate([csf_a, csf_b]):
                            for jCSF, csf_orig in enumerate(list_CSF):
                                h = Helm_between_CSFs(Enuc, obt, tbt, csf_orig, csf_ex)
                                Hmat_chain_aug[nCSF + k, jCSF] = h
                                Hmat_chain_aug[jCSF, nCSF + k] = h
                        n_eig = min(eigen, nCSF + 1)
                        evals_chain, evecs_chain = scipy.sparse.linalg.eigsh(csr_matrix(Hmat_chain_aug), k=n_eig, which='SA')
                        psi_chain = evecs_chain[:, np.argmin(evals_chain)]
                        s_a = np.sign(psi_chain[nCSF])   if abs(psi_chain[nCSF])   > 1e-10 else 1.0
                        s_b = np.sign(psi_chain[nCSF+1]) if abs(psi_chain[nCSF+1]) > 1e-10 else s_a
                        isign = round(s_a * s_b)
                        print(f' Chain pair {chain_group[0][0]} <-> {chain_group[1][0]}: '
                              f'psi[nCSF]={psi_chain[nCSF]:.6f}  psi[nCSF+1]={psi_chain[nCSF+1]:.6f}  isign={isign:+d}')

                        # Reference vector over the SHARED SD basis.
                        sym_CSF_vec = chain_vecs[_members[0]] + isign * chain_vecs[_members[1]]
                        _nv = np.linalg.norm(sym_CSF_vec)
                        if _nv > 1e-16:
                            sym_CSF_vec = sym_CSF_vec / _nv

                        # Explicit full-ex H basis (symadapted) — keep per-chain spaces.
                        ex_space_a, _ = build_chain_group_excitation_space(csf_a, _cg_sub_pairs, _cg_sub_map, _cg_n_eff, nparal)
                        ex_space_b, _ = build_chain_group_excitation_space(csf_b, _cg_sub_pairs, _cg_sub_map, _cg_n_eff, nparal)
                        symad_ex_space, _, _ = combine_two_list_ex_spaces(ex_space_a, ex_space_b, isign, False)

                        list_list_ex_space.append(shared_ex_sp)
                        list_list_symadapted_ex_space.append(symad_ex_space)
                        list_list_ia_pair.append(list(effective_ia_pairs))
                        list_list_genmat.append(shared_gm)
                        list_sym_CSF_vec.append(sym_CSF_vec)
                        list_list_extra_vecs.append([])
                        list_is_basis_ext_group.append(True)
                        list_list_frozen_positions.append([])
                        print(f'    Added degenerate chain basis-extension group (shared U): '
                              f'ref over {len(shared_ex_sp)} SDs')
                    else:
                        for _mi, (_, ref_csf) in zip(_members, chain_group):
                            print(f' Building the ex space for the excited CSF {ref_csf}')
                            print(f' Utilizing the ia pairs {_cg_sub_pairs}')
                            print(f' With coeffs {_cg_sub_map}')
                            sym_CSF_vec = chain_vecs[_mi].copy()
                            # Explicit full-ex H basis (symadapted) — per-chain space.
                            basis_ex_space, _ = build_chain_group_excitation_space(ref_csf, _cg_sub_pairs, _cg_sub_map, _cg_n_eff, nparal)
                            list_list_ex_space.append(shared_ex_sp)
                            list_list_symadapted_ex_space.append(list(basis_ex_space))
                            list_list_ia_pair.append(list(effective_ia_pairs))
                            list_list_genmat.append(shared_gm)
                            list_sym_CSF_vec.append(sym_CSF_vec)
                            list_list_extra_vecs.append([])
                            list_is_basis_ext_group.append(True)
                            list_list_frozen_positions.append([])
                            print(f' Added basis-extension group (shared U): ref over {len(shared_ex_sp)} SDs')

        _t_post1 = time.time()
        _t_csf1 = time.time()
        print(f'[TIMING] CSF{iCSF}: post-processing (sort + effective_ia_pairs + GS/genmat rebuild) '
              f'took {_t_post1 - _t_post0:.3f}s')
        print(f'[TIMING] CSF{iCSF}: TOTAL {_t_csf1 - _t_csf0:.3f}s')

        print(f'Updated length: {len(list_list_genmat)}')

    print(f'Parentage List:', list_is_basis_ext_group)

    print('\n=== CSF Group Parentage ===')
    for g in range(len(list_list_ex_space)):
        ref_idx = list_list_ex_space[g][0][1]
        ia_pairs = list_list_ia_pair[g]
        is_basis_ext = list_is_basis_ext_group[g]
        print(f'Group {g}: ref_SD_idx={ref_idx}, ia_pairs={ia_pairs}, is_basis_ext={is_basis_ext}')

    print('\n=== Shared Unitary Groups ===')
    parent_map = {}
    current_parent = None
    for g in range(len(list_list_ex_space)):
        if list_is_basis_ext_group[g] == False:
            current_parent = g
            parent_map[current_parent] = current_parent
        else:
            parent_map[g] = current_parent

    print('parent map:', parent_map)

    shared_unitary_groups = {}
    for g, parent in parent_map.items():
        if parent not in shared_unitary_groups:
            shared_unitary_groups[parent] = []
        shared_unitary_groups[parent].append(g)

    for parent, members in shared_unitary_groups.items():
        print(f' Shared unitary group (parent={parent}): members={members}')

    return list_list_ex_space, list_list_ia_pair, list_list_genmat, list_sym_CSF_vec, list_improvement, list_improvement_ia, list_list_extra_vecs, shared_unitary_groups, list_list_symadapted_ex_space, list_list_frozen_positions


def supplement_pipi_pair_ex(list_selected_ia,grouped_ia_pairs,debug=False):
    """
    if 5,6 is a pi set, 7, 8 is a pi set, (5->7,6->8) is selected, but (5->8,6->7) is not selected,
    supplemented (5->8,6->7). This is because the (5,6) empty and (7,8) fully occupied can come
    from both routes.
    """
    if debug: print('\nIn supplement_pipi_pair_ex')

    list_pair_to_add = []
    for ia_pair in list_selected_ia:
        if len(ia_pair) == 2:
            [[dmo1,vmo1],[dmo2,vmo2]] = ia_pair
            if dmo1 == dmo2 or vmo1 == vmo2: continue
            if [[dmo1,vmo2],[dmo2,vmo1]] in list_selected_ia or [[dmo2,vmo1],[dmo1,vmo2]] in list_selected_ia: continue
            if [[dmo1,vmo2],[dmo2,vmo1]] in grouped_ia_pairs:
                list_pair_to_add.append([[dmo1,vmo2],[dmo2,vmo1]])
            elif [[dmo2,vmo1],[dmo1,vmo2]] in grouped_ia_pairs:
                list_pair_to_add.append([[dmo2,vmo1],[dmo1,vmo2]])
            else:
                print(f'Strange! {ia_pair} in but {[[dmo2,vmo1],[dmo1,vmo2]]} \
                  or {[[dmo1,vmo2],[dmo2,vmo1]]} not in total list of pairs')
                sys.exit()

    if len(list_pair_to_add) != 0:
        print(f'The following pairs are to be added')
        print(list_pair_to_add)
        supp_list_selected_ia = list_selected_ia + list_pair_to_add
        print(f'Supplemented list of selected pairs: {list_selected_ia}')
        input("Press Enter to continue...")
    else:
        supp_list_selected_ia = list_selected_ia

    return supp_list_selected_ia

def make_list_genmat_for_sym_CSF(CSF1,CSF2,SOMO_DMO1,SOMO_DMO2,isym_sign,grouped_ia_pairs,list_theta,debug=False):
    """
    Prepare and combine list_genmats for symmetrized CSF of two sym partners CSF1 and 2.
    """

    if debug: print('\nIn make_list_genmat_for_sym_CSF')

    ungroupped_ia_pair = []
    for pairs in grouped_ia_pairs:
        for pair in pairs:
            ungroupped_ia_pair.append(pair)

    if debug: print(f'ungroupped_ia_pair: {ungroupped_ia_pair}')
    [onlist,onidx_list,coefvec] = CSF1
    list_pair_ex_space1, list_pair_ex_comb1 = make_pair_ex_space_fast(onlist,onidx_list,coefvec,ungroupped_ia_pair)
   #print('list_pair_ex_comb1:')
   #print(list_pair_ex_comb1)
   #print('list_pair_ex_space1:')
   #print(list_pair_ex_space1)
    [onlist,onidx_list,coefvec] = CSF2
    list_pair_ex_space2, list_pair_ex_comb2 = make_pair_ex_space_fast(onlist,onidx_list,coefvec,ungroupped_ia_pair)
   #print('list_pair_ex_space2:')
   #print(list_pair_ex_space2)
   #print('list_pair_ex_comb2:')
   #print(list_pair_ex_comb2)

    list_pair_ex_combine, list_corspnd, sym_CSF_vec = combine_two_list_ex_spaces(list_pair_ex_space1,list_pair_ex_space2,isym_sign,False)
    ndim = max(list_corspnd) + 1

   #print(f'list_corspnd: {list_corspnd}')
   #print(f'sym_CSF_vec: {sym_CSF_vec}')

    list_ungroupped_ia_pair = []
    for ia_pair in ungroupped_ia_pair:
        list_ungroupped_ia_pair.append([ia_pair])

    list_iapair_st_pairs1 = make_Hmat_diagonal_space(None,None,None,list_pair_ex_space1,list_pair_ex_comb1,False,False)
    list_iapair_st_pairs2 = make_Hmat_diagonal_space(None,None,None,list_pair_ex_space2,list_pair_ex_comb2,False,False)
    list_genmat1 = make_iapair_genmat_fast(list_ungroupped_ia_pair,list_pair_ex_space1,list_iapair_st_pairs1,False)
    list_genmat2 = make_iapair_genmat_fast(list_ungroupped_ia_pair,list_pair_ex_space2,list_iapair_st_pairs2,False)
   #print('list_iapair_st_pairs1:')
   #print(list_iapair_st_pairs1)
   #print('list_iapair_st_pairs2:')
   #print(list_iapair_st_pairs2)
   #print('list_genmat1:')
   #for genmat1 in list_genmat1:
   #    print(genmat1)
   #print('list_genmat2:')
   #for genmat2 in list_genmat2:
   #    print(genmat2)

    list_regroupped_ia_pairs = []

    list_genmat = []
    for ia_pair in grouped_ia_pairs:
        print(f'ia_pair in group: {ia_pair}, {len(ia_pair)}')
        if len(ia_pair) == 1:
           #print(ungroupped_ia_pair.index(ia_pair[0]),ia_pair)
            [pair] = ia_pair
            ind_ungroupped = ungroupped_ia_pair.index(pair)
            genmat1 = list_genmat1[ind_ungroupped]
            genmat2 = list_genmat2[ind_ungroupped]
            genmat = combine_genmats(genmat1,genmat2,list_corspnd,ndim)
            list_regroupped_ia_pairs.append(ia_pair)
            list_genmat.append(genmat)
        elif len(ia_pair) == 2:
            [[dmo1,vmo1],[dmo2,vmo2]] = ia_pair
           #print(dmo1,vmo1,dmo2,vmo2)
           #print(ungroupped_ia_pair.index([dmo1,vmo1]))
           #print(ungroupped_ia_pair.index([dmo2,vmo2]))
            genmat1_1 = list_genmat1[ungroupped_ia_pair.index([dmo1,vmo1])]
            genmat1_2 = list_genmat1[ungroupped_ia_pair.index([dmo2,vmo2])]
            genmat2_1 = list_genmat2[ungroupped_ia_pair.index([dmo1,vmo1])]
            genmat2_2 = list_genmat2[ungroupped_ia_pair.index([dmo2,vmo2])]
            norm1_1 = scipy.sparse.linalg.norm(genmat1_1)
            norm1_2 = scipy.sparse.linalg.norm(genmat1_2)
            norm2_1 = scipy.sparse.linalg.norm(genmat2_1)
            norm2_2 = scipy.sparse.linalg.norm(genmat2_2)
           #print(ia_pair)
           #print('genmat1_1:')
           #print(genmat1_1)
           #print('genmat1_2:')
           #print(genmat1_2)
           #print('genmat2_1:')
           #print(genmat2_1)
           #print('genmat2_2:')
           #print(genmat2_2)
            assert norm1_1 == norm2_2 and norm1_2 == norm2_1
            if np.isclose(norm1_1,0.0) and np.isclose(norm1_2,0.0) and np.isclose(norm2_1,0.0) and np.isclose(norm2_2,0.0):
               #print('Strange: both norm1_1 and norm1_2 = 0. If so, the ia_pair shall not be included')
                print('Strange: all four norms = 0. If so, the ia_pair shall not be included')
                print(ia_pair,SOMO_DMO1,SOMO_DMO2)
                print('Bombing out!')
                sys.exit()
            genmat_11_22 = combine_genmats(genmat1_1,genmat2_2,list_corspnd,ndim)
            genmat_12_21 = combine_genmats(genmat1_2,genmat2_1,list_corspnd,ndim)

            if not np.isclose(scipy.sparse.linalg.norm(genmat_11_22),0.0):
                list_regroupped_ia_pairs.append(ia_pair)
                list_genmat.append(genmat_11_22)
            if not np.isclose(scipy.sparse.linalg.norm(genmat_12_21),0.0):
                list_regroupped_ia_pairs.append(ia_pair)
                list_genmat.append(genmat_12_21)


        elif len(ia_pair) == 4:
            print(f'Not yet coded for group pairs length = 4. Bombing out!')
            sys.exit()
        else:
            print(f'Length of pairs shall be within 1, 2, 4 only. {ia_pair}. Bombing out!')
            sys.exit()

    if debug:
        print(f'regroupped ia pairs: {list_regroupped_ia_pairs}')
        print('combined list_genmat:')
        for genmat in list_genmat:
            print(genmat)

    return list_regroupped_ia_pairs, list_genmat, list_pair_ex_combine, sym_CSF_vec

def combine_list_genmat(list_genmat1,list_genmat2,list_corspnd):
    """
    Combine the genmats of the two sym-paired CSFs.
    """

    ndim = max(list_corspnd) + 1
   #print(f'ndim in combine_list_genmat: {ndim}')
    assert len(list_genmat1) == len(list_genmat2)
    list_genmat = []
    for ii in range(len(list_genmat1)):
        genmat = combine_genmats(list_genmat1[ii],list_genmat2[ii],list_corspnd,ndim)
        list_genmat.append(genmat)

    return list_genmat

def transform_genmat_combine_basis(G, ep0, ep1, isign):
    """
    Transform genmat G when two ex_space basis states are combined:
        new_ep0 = (old_ep0 + isign * old_ep1) / sqrt(2)
        old_ep1 is deleted.
    Returns a dense ndarray of shape (ndim-1, ndim-1).
    """
    if scipy.sparse.issparse(G):
        G = G.toarray()
    G = np.array(G, dtype=float)
    ndim = G.shape[0]
    # combine rows
    G[ep0, :] = (G[ep0, :] + isign * G[ep1, :]) / np.sqrt(2)
    # combine cols
    G[:, ep0] = (G[:, ep0] + isign * G[:, ep1]) / np.sqrt(2)
    # delete ep1 row and col
    G = np.delete(G, ep1, axis=0)
    G = np.delete(G, ep1, axis=1)
    return G

def combine_genmats(genmat1,genmat2,list_corspnd,ndim):
    """
    Combine two genmats
    """

    genmat = csr_matrix((ndim,ndim))
    ndim1 = genmat1.shape[0]
    assert ndim1 < ndim
   #print(genmat.shape,genmat1.shape)
    genmat[0:ndim1,0:ndim1] = genmat1

    row, col = genmat2.nonzero()
   #print(row)
   #print(col)
    for ii in range(len(row)):
       irow = row[ii]
       icol = col[ii]
       irow_match = list_corspnd[irow]
       icol_match = list_corspnd[icol]
       genmat[irow_match,icol_match] = genmat2[irow,icol]

   #Makre sure genmat is antisymmetric
    assert np.isclose(scipy.sparse.linalg.norm(genmat + genmat.transpose()),0.0)

    return genmat

def combine_two_list_ex_spaces(list_ex_space1,list_ex_space2,isign,debug=False):
    """
    Combine two list_ex_spaces to one and also output the indices correlation matrix
    """

    if debug: print('\nIn combine_two_list_ex_spaces')

    list_ex_space = copy.deepcopy(list_ex_space1)
    list_corspnd = []
    for iCSF2, CSF in enumerate(list_ex_space2):
        lfound = False
        for ii, CSF1 in enumerate(list_ex_space1):
            Selm = overlap_CSFs(CSF,CSF1)
            if not np.isclose(Selm,0.0):
                if not np.isclose(abs(Selm),1.0):
                    print('Strange! Fractional overlap is detected. Bombing out!')
                    print('CSF:')
                    print(CSF)
                    print('CSF1:')
                    print(CSF1)
                    sys.exit()
                if np.isclose(Selm,-1.0):
                    print(f'Strange! -1 overlap detected between {CSF} and {CSF1}. Bombing out!')
                    sys.exit()
                lfound = True
                if iCSF2 == 0 and lfound:
                    print(f'The 0th CSF in list_ex_space2 is found in list_ex_space2. This is unreasonable. Bombing out!')
                    sys.exit()
                list_corspnd.append(ii)
                break
        if not lfound:
            list_ex_space.append(CSF)
            list_corspnd.append(len(list_ex_space)-1)

    sym_CSF_vec = np.zeros(len(list_ex_space))

    sym_CSF_vec[0] = np.sqrt(0.5)
    sym_CSF_vec[len(list_ex_space1)] = np.sqrt(0.5)
    if isign < 0: sym_CSF_vec[len(list_ex_space1)] = -np.sqrt(0.5)

    return list_ex_space,list_corspnd, sym_CSF_vec

def check_combined_list_genmat(CSF1,CSF2,isign,list_pair_ex_combine,list_genmat,sym_CSF_vec,list_ia_pair):
    """
    Check the correctness of the combined list_genmat from individual list_genmat
    """
    print('\nIn check_combined_list_genmat')

    coefs = np.array([np.sqrt(0.5),np.sqrt(0.5)])
    if isign < 0: coefs[1] = -coefs[1]

    comb_CSF = LC_CSFs([CSF1,CSF2],coefs,l_check_norm=True)

    print(len(list_genmat),len(list_ia_pair))
    for ipair, ia_pair in enumerate(list_ia_pair):
        T_op = FermionOperator()
        for [ii,aa] in ia_pair:
            T_op += get_Tiiaa_00(ii,aa)

        T_op = normal_ordered(T_op)
        T_op.compress()
        onlist, onidx_list, on_coefs = op_action_tz_remove_0coef(T_op,comb_CSF[0],comb_CSF[1],comb_CSF[2])
        T_op_CSF = [onlist, onidx_list, on_coefs]
        T_op_vec = list_genmat[ipair]@sym_CSF_vec
        T_op_vec_CSF = LC_CSFs(list_pair_ex_combine,T_op_vec)
        norm_T_op_CSF = np.linalg.norm(T_op_CSF[2])
        norm_T_op_vec_CSF = np.linalg.norm(T_op_vec_CSF[2])
        if np.isclose(norm_T_op_CSF,0.0) and np.isclose(norm_T_op_vec_CSF,0.0):
            print('Good! Both result in null CSF, as expected. Test passed.')
            continue
        assert np.isclose(norm_T_op_CSF,norm_T_op_vec_CSF)
        Selm = overlap_CSFs(T_op_CSF,T_op_vec_CSF)
        Selm /= norm_T_op_CSF*norm_T_op_vec_CSF
        print(f'Selm: {Selm}')
        if not np.isclose(Selm,1.0):
            print(f'ia_pair: {ia_pair}')
            print('CSF1:')
            print(CSF1)
            print('CSF2:')
            print(CSF2)
            print('comb_CSF:')
            print(comb_CSF)
            print('T_op_CSF')
            print(T_op_CSF)
            print('sym_CSF_vec:')
            print(sym_CSF_vec)
            print('T_op_vec:')
            print(T_op_vec)
            print('genmat:')
            print(list_genmat[ipair])
            print('T_op_vec:')
            print(T_op_vec)
            sys.exit()

def pick_pairex_within_CAS(list_list_pair_ex_space,list_sym_CSF_vec,actmo_start,actmo_end,l_axial_sym,Enuc,obt,tbt,nparal=1,list_mo_exclud=[],debug=False):
    """
    Within each pair_ex_space, pick out the states that are obtained from the 0th CSF
    by pair excitation within CAS
    """

    if debug: print('\nIn pick_pairex_within_CAS')


    list_list_pairex_CSF_vec = []
    list_list_pairex_CSF = []
    n_total_CSF_in_CAS = 0
    for irefCSF, list_pair_ex_space in enumerate(list_list_pair_ex_space):
        sym_CSF_vec = list_sym_CSF_vec[irefCSF]
        ind_ref_CSF = np.where(sym_CSF_vec != 0.0)[0]
        ind_ex_CSF = np.where(sym_CSF_vec == 0.0)[0]
        list_set_dmo_ref = []
        for ii in ind_ref_CSF:
            list_set_dmo_ref.append(set(dmo_in_SD(list_pair_ex_space[ii][0][0])))

       #print(ii,ind_ref_CSF,ind_ex_CSF)
        print(f'\n{ii,list_set_dmo_ref}')
        ind_ex_from_ref = []
        for iex in ind_ex_CSF:
            set_dmo_exCSF = set(dmo_in_SD(list_pair_ex_space[iex][0][0]))
            for jref, set_dmo_ref in enumerate(list_set_dmo_ref):
                dmo_diff = set_dmo_ref^set_dmo_exCSF
               #print(iex,jref,set_dmo_ref,set_dmo_exCSF,dmo_diff)
                l_within_CAS = True
                for mo in dmo_diff:
                    if mo > actmo_end or mo < actmo_start or mo in list_mo_exclud:
                        l_within_CAS = False
                        break
                if not l_within_CAS: continue
                if iex in ind_ex_from_ref: continue
               #print('This CSF is connected to ref CSF(s) by pair excitation within CAS')
                ind_ex_from_ref.append(iex)

        list_pairex_CSF_vec = []
        ndim = len(list_pair_ex_space)
        vec0 = np.zeros(ndim)
        if len(ind_ex_from_ref) != 0:
            print('\nThe following CSFs are obtained from ref CSF(s) by pair excitations within CAS')
            list_exCSF_E = []
            for iex in ind_ex_from_ref:
                E_CSF = Helm_between_CSFs(Enuc,obt,tbt,list_pair_ex_space[iex],list_pair_ex_space[iex])
                list_exCSF_E.append(E_CSF)
                print(iex,dmo_in_SD(list_pair_ex_space[iex][0][0]),E_CSF)

#zip(*sorted(zip(list_improve,list_ia_one_CSF)))

            if l_axial_sym:
                list_refCSFs = []
                coefs_ref = []
                for iref in ind_ref_CSF:
                    list_refCSFs.append(list_pair_ex_space[iref])
                    coefs_ref.append(sym_CSF_vec[iref])

                coefs_ref = np.array(coefs_ref)
                sym_ref_CSF = LC_CSFs(list_refCSFs,coefs_ref)
               #sorted_ex_CSF_E, sorted_ind_ex = zip(*sorted(zip(list_exCSF_E,ind_ex_from_ref)))
               #sorted_ex_CSF_E = list(sorted_ex_CSF_E)
               #sorted_ind_ex = list(sorted_ind_ex)
               #print('\nE-sorted exCSFs:')
               #for iex_ind in range(len(sorted_ind_ex)):
               #    print(sorted_ind_ex[iex_ind],sorted_ex_CSF_E[iex_ind])
                list_ref_ex_CSFs = [sym_ref_CSF]
                for iex in ind_ex_from_ref:
                    list_ref_ex_CSFs.append(list_pair_ex_space[iex])

                if nparal > 1:
                    Hmat = construct_Hmat_CSFs_paral_triu(list_ref_ex_CSFs,Enuc,obt,tbt,nparal)
                else:
                    Hmat = construct_Hmat_CSFs(list_ref_ex_CSFs,Enuc,obt,tbt)
                print('\nHamiltonian matrix of ref CSF and those obtained from pair excitations\n')
                print_matrix(Hmat)
               #Hmat_sparse = csr_matrix(Hmat)
               #_, psi_GS = get_ground_state(Hmat_sparse)
               #print(f'\nGround state eigenvector: {psi_GS}')
                degen_thrsh = 1.0e-7
                list_pairing = []
                for iex_bas in range(1,Hmat.shape[0]-1):
                    iex = iex_bas - 1
                    l_paired = False
                    for item in list_pairing:
                        if iex in item:
                            l_paired = True
                            break
                    if l_paired: continue
                    l_pairing = False
                    for jex_bas in range(iex_bas+1,Hmat.shape[0]):
                        jex = jex_bas - 1
                        if abs(Hmat[iex_bas,iex_bas] - Hmat[jex_bas,jex_bas]) < degen_thrsh:
                            list_pairing.append([iex,jex])
                            l_pairing = True
                            break
                    if not l_pairing: list_pairing.append([iex])
                iex = len(ind_ex_from_ref)-1
                l_paired = False
                for item in list_pairing:
                    if iex in item: l_paired = True
                if not l_paired: list_pairing.append([iex])
                print('\nPairing of the exCSFs based on their degeneracies:')
                print(list_pairing)

                for item in list_pairing:
                    if len(item) == 1:
                        iex_ind = ind_ex_from_ref[item[0]]
                        ex_CSF_vec = copy.deepcopy(vec0)
                        ex_CSF_vec[iex_ind] = 1.0
                       #print(f'Adding 1.0 to {iex_ind}')
                    elif len(item) == 2:
                        [iex,jex] = item
                        iex_bas,jex_bas = iex+1, jex+1
                       #isign = round(np.sign(psi_GS[iex_bas]) * np.sign(psi_GS[jex_bas]))
                        assert np.isclose(abs(Hmat[0,iex_bas]),abs(Hmat[0,jex_bas]))
                        isign = round(np.sign(Hmat[0,iex_bas]) * np.sign(Hmat[0,jex_bas]))
                        iex_ind = ind_ex_from_ref[iex]
                        jex_ind = ind_ex_from_ref[jex]
                        ex_CSF_vec = copy.deepcopy(vec0)
                        ex_CSF_vec[iex_ind], ex_CSF_vec[jex_ind] = np.sqrt(0.5), np.sqrt(0.5)
                        if isign < 0: ex_CSF_vec[jex_ind] *= -1.0
                       #print(f'Adding sqrt(0.5) to {iex_ind,jex_ind}, {ex_CSF_vec[iex_ind],ex_CSF_vec[jex_ind]}')
                    else:
                        print(f'So far, only double degeneracy is supported. {item}. Bombing out!')
                        sys.exit()

                    list_pairex_CSF_vec.append(ex_CSF_vec)

                print('\nCollected ex_CSF_vec:')
                for item in list_pairex_CSF_vec: print(csr_matrix(item))


            else:
                for iex in ind_ex_from_ref:
                    ex_CSF_vec = copy.deepcopy(vec0)
                    ex_CSF_vec[iex] = 1.0
                    list_pairex_CSF_vec.append(ex_CSF_vec)

                print('\nCollected ex_CSF_vec:')
                for item in list_pairex_CSF_vec: print(csr_matrix(item))

        list_pairex_CSF_vec = [sym_CSF_vec] + list_pairex_CSF_vec
        list_list_pairex_CSF_vec.append(list_pairex_CSF_vec)
       #Get the true pairex CSFs for future use, not just the vectors
        list_pairex_CSF = []
        for CSFvec in list_pairex_CSF_vec:
            #assert len(CSFvec) == len(list_pair_ex_space)
            list_pairex_CSF.append(LC_CSFs(list_pair_ex_space,CSFvec))

        assert len(list_pairex_CSF) == len(list_pairex_CSF_vec)
        n_total_CSF_in_CAS += len(list_pairex_CSF)

        list_list_pairex_CSF.append(list_pairex_CSF)
        print(f'Dimension reduction for CSF Group {irefCSF} from {len(list_pair_ex_space)} to {len(list_pairex_CSF)}')

    assert len(list_list_pairex_CSF_vec) == len(list_list_pairex_CSF)
    print(f'\nTotal number of CSFs in CAS leftover {n_total_CSF_in_CAS}')

    return list_list_pairex_CSF_vec, list_list_pairex_CSF

def remove_pairex_in_CAS(actmo_start,actmo_end,list_list_ia,list_list_genmat,list_list_theta,l_use_decomp_genmat=False,list_list_decomp_genmat=[]):
    """
    Remove the ia pair between active orbitals
    """

    print('\nIn remove_pairex_in_CAS')

    assert len(list_list_ia) == len(list_list_genmat)
    assert len(list_list_genmat) == len(list_list_theta)
    if l_use_decomp_genmat: assert len(list_list_theta) == len(list_list_decomp_genmat)

    list_list_ia_noCAS = []
    list_list_genmat_noCAS = []
    list_list_theta_noCAS = []
    list_list_decomp_genmat_noCAS = []
    for iref_CSF in range(len(list_list_ia)):
        print(f'Removing inCAS Ex for CSF group {iref_CSF}')
        list_ia_noCAS = copy.deepcopy(list_list_ia[iref_CSF])
        list_genmat_noCAS = copy.deepcopy(list_list_genmat[iref_CSF])
        list_theta_noCAS = copy.deepcopy(list_list_theta[iref_CSF])
        if l_use_decomp_genmat: list_decomp_genmat_noCAS = copy.deepcopy(list_list_decomp_genmat[iref_CSF])

        list_l_remove = [False] * len(list_ia_noCAS)
        for ipairs in range(len(list_ia_noCAS)):
            pairs = list_ia_noCAS[ipairs]
            for pair in pairs:
                if pair[0] >= actmo_start and pair[0] <= actmo_end and \
                   pair[1] >= actmo_start and pair[1] <= actmo_end:
                    list_l_remove[ipairs] = True
                    break
            if list_l_remove[ipairs]: print(f'To remove {pairs}')

        for  ipairs in range(len(list_ia_noCAS)-1,-1,-1):
            if list_l_remove[ipairs]:
                del list_ia_noCAS[ipairs], list_genmat_noCAS[ipairs], list_theta_noCAS[ipairs]
                if l_use_decomp_genmat: del list_decomp_genmat_noCAS[ipairs]

        print(f'Resulting external pair ex for CSF group {iref_CSF}: {list_ia_noCAS}')
        assert len(list_ia_noCAS) == len(list_genmat_noCAS)
        assert len(list_ia_noCAS) == len(list_theta_noCAS)
        if l_use_decomp_genmat: assert len(list_ia_noCAS) == len(list_decomp_genmat_noCAS)
        list_list_ia_noCAS.append(list_ia_noCAS)
        list_list_genmat_noCAS.append(list_genmat_noCAS)
        list_list_theta_noCAS.append(list_theta_noCAS)
        if l_use_decomp_genmat: list_list_decomp_genmat_noCAS.append(list_decomp_genmat_noCAS)


    return list_list_ia_noCAS, list_list_genmat_noCAS, list_list_theta_noCAS, list_list_decomp_genmat_noCAS

def select_ia_pairs_AVO_1by1(list_CSF,group_mp2_iapair,group_mp2_ampld,Enuc,obt,tbt,Ethrsh=1.0e-3,nparal=1,debug=False):
    """
    Select ia pairs for CSFs based on 1 by 1 ia pairs AVO scheme for the energy of <CSF|U^+ H U|CSF>
    """

    if debug: print('\nIn select_ia_pairs_AVO_1by1')

   #At this time the group_mp2_ia_pair shall only contain one ia pair in each group
    for group in group_mp2_iapair:
        assert len(group) == 1

    list_UCSF_opt = []
    list_list_ia_pair_UCSF = []
    list_list_theta_UCSF = []
    list_list_pair_ex_space = []
    list_list_group_pairs = []
    list_list_genmat = []
    list_sym_CSF_vec = []
    for iCSF, CSF in enumerate(list_CSF):
        print(f'\noptimizing U for CSF{iCSF} in AVO 1by1')
        pool_iapair = copy.deepcopy(group_mp2_iapair)
        UCSF_old = copy.deepcopy(CSF)
        E_old = Helm_between_CSFs(Enuc,obt,tbt,UCSF_old,UCSF_old)
        list_ia_for_one_UCSF = []
        while(len(pool_iapair) != 0):
            list_of_pairs_with_improve = []
            list_of_improve = []
            list_UCSF_opt_1ia = []
            list_one_ia_theta = []
            for ipair, pair in enumerate(pool_iapair):
                list_one_ampld = [group_mp2_ampld[group_mp2_iapair.index(pair)]]
                E_UCSF_mp2, UCSF_mp2 = E_UCSF_manual(UCSF_old,[pair],list_one_ampld,Enuc,obt,tbt)
                E_UCSF_old = Helm_between_CSFs(Enuc,obt,tbt,UCSF_old,UCSF_old)
                if np.isclose(E_UCSF_mp2,E_UCSF_old):
                    continue
                Selm = overlap_CSFs(UCSF_old,UCSF_mp2)
                if np.isclose(Selm,1.0):
                    continue
                UCSF_opt, vec_theta_opt, improve = opt_E_UCSF_manual(UCSF_old,[pair],list_one_ampld,Enuc,obt,tbt)
               #This improve is the improvement from the initial guess of U mp2, not the improve from
               #UCSF_old. improve has to be re-defined
                improve = Helm_between_CSFs(Enuc,obt,tbt,UCSF_opt,UCSF_opt) - E_UCSF_old
                if abs(improve) < Ethrsh:
                    continue
                list_of_pairs_with_improve.append(pair)
                list_of_improve.append(improve)
                list_one_ia_theta.append(vec_theta_opt[0])

            if len(list_of_pairs_with_improve) == 0:
                UCSF_new = UCSF_old
                break
            sorted_list_of_improve, sorted_list_of_pairs, sorted_list_thetas =\
              zip(*(sorted(zip(list_of_improve,list_of_pairs_with_improve,list_one_ia_theta))))
            if debug:
                print(sorted_list_of_improve)
                print(sorted_list_of_pairs)
                print(sorted_list_thetas)
            ia_pair_max = sorted_list_of_pairs[0]
           #print(ia_pair_max,sorted_list_of_improve[0],sorted_list_thetas[0])
            UCSF_new = prepare_UCSF_one_CSF_manual(UCSF_old,[ia_pair_max],[sorted_list_thetas[0]])
            E_UCSF_new = Helm_between_CSFs(Enuc,obt,tbt,UCSF_new,UCSF_new)
            if not np.isclose((E_UCSF_new - E_UCSF_old), sorted_list_of_improve[0]):
                print(f'E UCSF_new {E_UCSF_new}, diff. from E_UCSF_old {E_UCSF_new - E_UCSF_old}')
                sys.exit()
            list_ia_for_one_UCSF.append([ia_pair_max,sorted_list_thetas[0],sorted_list_of_improve[0]])
            pool_iapair = list(sorted_list_of_pairs)
            pool_iapair.remove(ia_pair_max)
            if debug:
                print(f'Leftover ia pairs with dimension: {len(pool_iapair)}')
                print(pool_iapair)

            UCSF_old = UCSF_new

        print(f'\nia pairs selected for CSF{iCSF}')
        list_ia_pair_one_UCSF = []
        list_theta_one_UCSF = []
        list_group_pairs = []
        for item in list_ia_for_one_UCSF:
            print(item)
            list_ia_pair_one_UCSF.append(item[0][0])
            list_theta_one_UCSF.append(item[1])
            list_group_pairs.append([item[0][0]])

        UCSF_final = prepare_UCSF_one_CSF_manual(CSF,list_group_pairs,list_theta_one_UCSF)
        Selm = overlap_CSFs(UCSF_final,UCSF_new)
        assert np.isclose(Selm,1.0)
        list_UCSF_opt.append(UCSF_final)
        list_list_ia_pair_UCSF.append(list_ia_pair_one_UCSF)
        list_list_theta_UCSF.append(list_theta_one_UCSF)

       #Also prepare genmats and list_pair_ex_space
        UCSF_matrix, Uvec, list_pair_ex_space, Umat, list_genmat =\
          prepare_UCSF_for_one_CSF(CSF,list_group_pairs,list_theta_one_UCSF,nparal,False)
        Selm = overlap_CSFs(UCSF_final,UCSF_matrix)
        if not np.isclose(Selm,1.0):
            Selm_final = overlap_CSFs(UCSF_final,UCSF_final)
            Selm_matrix = overlap_CSFs(UCSF_matrix,UCSF_matrix)
            print(f'Selm: {Selm} not close to 1, Selm_final: {Selm_final}, Selm_matrix: {Selm_matrix}')
            sorted_UCSF_final_coefs, sorted_UCSF_final_idx = zip(*sorted(zip(UCSF_final[2],UCSF_final[1])))
            sorted_UCSF_matrix_coefs, sorted_UCSF_matrix_idx = zip(*sorted(zip(UCSF_matrix[2],UCSF_matrix[1])))
            print(f'Dimensions of final and matrix: {len(sorted_UCSF_final_coefs),len(sorted_UCSF_matrix_coefs)}')
           #for ii in range(len(sorted_UCSF_final_coefs)):
           #    print(sorted_UCSF_final_coefs[ii],sorted_UCSF_final_idx[ii],sorted_UCSF_matrix_coefs[ii],sorted_UCSF_matrix_idx[ii])
            exCSFs_in_final_not_in_matrix = list(set(UCSF_final[1]).difference(set(UCSF_matrix[1])))
            n_spinorb = len(UCSF_final[0][0])
            print('CSFs in final not in matrix:')
            for item in exCSFs_in_final_not_in_matrix:
                print(item,get_on_vec(item,n_spinorb))
            sys.exit()
        sym_CSF_vec = np.zeros(len(list_pair_ex_space))
        sym_CSF_vec[0] = 1.0
        list_sym_CSF_vec.append(sym_CSF_vec)
        list_list_group_pairs.append(list_group_pairs)
        list_list_genmat.append(list_genmat)
        list_list_pair_ex_space.append(list_pair_ex_space)

    Hmat_CSF = construct_Hmat_CSFs_paral_triu(list_CSF,Enuc,obt,tbt,nparal)
    E_GS_CSF, psi_GS_CSF = get_ground_state(csr_matrix(Hmat_CSF))
    Hmat_UCSF_opt = construct_Hmat_CSFs_paral_triu(list_UCSF_opt,Enuc,obt,tbt,nparal)
    E_GS_UCSF_opt, psi_GS_UCSF_opt = get_ground_state(csr_matrix(Hmat_UCSF_opt))
    print(f'E0 of opt UCSFs: {E_GS_UCSF_opt}')
    print(f'Lowering of E0 brought by U: {E_GS_UCSF_opt-E_GS_CSF}')

    return list_list_pair_ex_space, list_list_group_pairs, list_list_genmat, list_list_theta_UCSF, list_sym_CSF_vec, list_UCSF_opt

def separate_ia_pairs_internal_external(list_list_ia,actmo_start,actmo_end,list_mo_exclud=[],debug=False):
    """
    Separate the ia pairs into internal (within active space) and external (the rest) sets
    """

    list_list_ia_internal = []
    list_list_ia_external = []

    for iref, list_ia in enumerate(list_list_ia):
        if debug: print(f'CSF group {iref}: list_ia: {list_ia}')
        list_ia_internal = []
        list_ia_external = []
        for ia_pairs in list_ia:
            l_internal = True
            for pair in ia_pairs:
                if min(pair) < actmo_start or max(pair) > actmo_end or bool(set(pair) & set(list_mo_exclud)): l_internal = False

            if l_internal:
                list_ia_internal.append(ia_pairs)
            else:
                list_ia_external.append(ia_pairs)

       #For excitation between degenerate shells, there may be duplicate ia_pairs.
       #Duplicate pairs are removed
        list_l_remove = [False] * len(list_ia_internal)
        for ipair in range(len(list_ia_internal)):
            for jpair in range(ipair+1,len(list_ia_internal)):
                if list_ia_internal[jpair] == list_ia_internal[ipair]: list_l_remove[jpair] = True
        for ipair in range(len(list_ia_internal)-1,-1,-1):
            if list_l_remove[ipair]: del list_ia_internal[ipair]

        list_l_remove = [False] * len(list_ia_external)
        for ipair in range(len(list_ia_external)):
            for jpair in range(ipair+1,len(list_ia_external)):
                if list_ia_external[jpair] == list_ia_external[ipair]: list_l_remove[jpair] = True
        for ipair in range(len(list_ia_external)-1,-1,-1):
            if list_l_remove[ipair]: del list_ia_external[ipair]
        if debug:
            print(f'\nFor refCSF group {iref}:')
            print(f'internal ia pairs: {list_ia_internal}')
            print(f'external ia pairs: {list_ia_external}')

        list_list_ia_internal.append(list_ia_internal)
        list_list_ia_external.append(list_ia_external)

    return list_list_ia_internal, list_list_ia_external

def print_CSF(CSF):
    for iSD, SD in enumerate(CSF[0]):
        print(SD,CSF[1][iSD],CSF[2][iSD])

def LC_CSFs(list_CSF,coefs,l_check_norm=False,l_combine_duplic_SD=True):
    """
    Linearly combine a list of CSFs using the input coefficients and return the resultant CSF
    """

   #print(coefs)

    assert len(coefs) == len(list_CSF)

   #print(f'In LC_CSFs:')
   #for item in list_CSF:
   #    print_CSF(item)

    onlist_res = []
    idx_list_res = []
    coefs_res = []
    for ii in range(len(coefs)):
        CSF = copy.deepcopy(list_CSF[ii])
       #CSF[2] *= coefs[ii]
        coefs_scaled = CSF[2]*coefs[ii]
        onlist_res += CSF[0]
        idx_list_res += CSF[1]
       #coefs_res += CSF[2].tolist()
        coefs_res += coefs_scaled.tolist()

    coefs_res = np.array(coefs_res)
    res_CSF = [onlist_res,idx_list_res,coefs_res]
   #Check normality
    if l_check_norm:
        Selm = overlap_CSFs(res_CSF,res_CSF)
        assert np.isclose(Selm,1.0)

   #Use the operation of identity operator to combine duplicate SDs resulting
   #from the linear combination of CSFs
    if l_combine_duplic_SD:
        op = FermionOperator.identity()
        onlist_post, idx_list_post, coefs_post = op_action_tz_remove_0coef(op,res_CSF[0],res_CSF[1],res_CSF[2])
        res_CSF = [onlist_post, idx_list_post, coefs_post]

    return res_CSF

def LC_CSFs_paral(list_CSF,coefs,nparal=1,l_check_norm=False,l_combine_duplic_SD=True,debug=False):
    """
    Linearly combine a list of CSFs using the input coefficients and return the resultant CSF
    This parallel version turns out slower than LC_CSFs. Not used!
    """

   #print(coefs)

    tic = time.perf_counter()

    assert len(coefs) == len(list_CSF)

   #print(f'In LC_CSFs:')
   #for item in list_CSF:
   #    print_CSF(item)

    onlist_res = []
    idx_list_res = []
    coefs_res = []
    for ii in range(len(coefs)):
        CSF = copy.deepcopy(list_CSF[ii])
       #CSF[2] *= coefs[ii]
        coefs_scaled = CSF[2]*coefs[ii]
        onlist_res += CSF[0]
        idx_list_res += CSF[1]
       #coefs_res += CSF[2].tolist()
        coefs_res += coefs_scaled.tolist()

    coefs_res = np.array(coefs_res)
    res_CSF = [onlist_res,idx_list_res,coefs_res]
   #Check normality
    if l_check_norm:
        Selm = overlap_CSFs(res_CSF,res_CSF)
        assert np.isclose(Selm,1.0)

    toc = time.perf_counter()
    print(f'Time before removing duplicate SDs: {toc - tic}')
    tic = toc
   #Use the operation of identity operator to combine duplicate SDs resulting
   #from the linear combination of CSFs
    if l_combine_duplic_SD and len(res_CSF[0]) > 0:
       #op = FermionOperator.identity()
       #res_CSF = op_action_tz_CSF_in_chunks(op,res_CSF,nparal)

        list_unique_idx = list(set(res_CSF[1]))
        n_comp = len(list_unique_idx)
        chunks = even_distrib_ncomp_to_nparal(n_comp,nparal)
        nchunk = len(chunks)
        list_chunks = []
        for ii in range(nchunk):
            list_chunks.append(list_unique_idx[chunks[ii][0]:chunks[ii][-1]+1])

       #print(f'res_CSF[1]: {res_CSF[1]}')
       #print(f'list_unique_idx: {list_unique_idx}')
       #print(f'list_chunks: {list_chunks}')

        def combine_on_in_chunk(list_idx,CSF_duplic_on):
            on_list_reduc = []
            list_idx_reduc = []
            list_coef_reduc = []
            for idx in list_idx:
                list_idx_pos = np.where(np.array(CSF_duplic_on[1]) == idx)[0]
                dsum = np.sum(CSF_duplic_on[2][list_idx_pos])
                onvec = CSF_duplic_on[0][list_idx_pos[0]]
                on_list_reduc.append(onvec)
                list_idx_reduc.append(idx)
                list_coef_reduc.append(dsum)

            CSF_chunk = [on_list_reduc,list_idx_reduc,np.array(list_coef_reduc)]
            return CSF_chunk
       #list_CSF_chunk = []
       #for chunk in list_chunks:
       #  ##print(f'chunk: {chunk}')
       #  # on_list_reduc = []
       #  # list_idx_reduc = []
       #  # list_coef_reduc = []
       #  # for idx in chunk:
       #  #     list_idx_pos = np.where(np.array(res_CSF[1]) == idx)[0]
       #  #    #print(f'list_idx_pos: {list_idx_pos}')
       #  #    #print(f'res_CSF[2]: {res_CSF[2]}')
       #  #     dsum = np.sum(res_CSF[2][list_idx_pos])
       #  #     onvec = res_CSF[0][list_idx_pos[0]]
       #  #     on_list_reduc.append(onvec)
       #  #     list_idx_reduc.append(idx)
       #  #     list_coef_reduc.append(dsum)
       #  #    #print(idx,dsum)

       #  # CSF_chunk = [on_list_reduc,list_idx_reduc,np.array(list_coef_reduc)]
       #    CSF_chunk = combine_on_in_chunk(chunk,res_CSF)
       #    list_CSF_chunk.append(CSF_chunk)

        list_res_CSF = [res_CSF]*nchunk
       #list_CSF_chunk = []
       #for ii in range(nchunk):
       #    CSF_chunk = CSF_chunk = combine_on_in_chunk(list_chunks[ii],list_res_CSF[ii])
       #    list_CSF_chunk.append(CSF_chunk)
        list_CSF_chunk = Parallel(n_jobs=nchunk)(delayed(combine_on_in_chunk)(list_chunks[ii],list_res_CSF[ii]) for ii in range(nchunk))
        toc = time.perf_counter()
        print(f'Time to remove duplicate SDs: {toc - tic}')
        tic = toc

       #coefs_all1 = np.ones([len(list_CSF_chunk)])
       #res_CSF = LC_CSFs(list_CSF_chunk,coefs_all1,l_combine_duplic_SD=False)
       #assert identical_CSFs(res_CSF_test_1,res_CSF)
        res_onlist = list_CSF_chunk[0][0]
        res_idxlist = list_CSF_chunk[0][1]
        res_coefs = list_CSF_chunk[0][2]
        for item in list_CSF_chunk[1:]:
            res_onlist += item[0]
            res_idxlist += item[1]
            res_coefs = np.hstack((res_coefs,item[2]))

        res_CSF = [res_onlist,res_idxlist,res_coefs]
        toc = time.perf_counter()
        print(f'Time to combine chunks: {toc - tic}')
        tic = toc



    if debug:
        res_CSF_test = LC_CSFs(list_CSF,coefs)
        if not identical_CSFs(res_CSF_test,res_CSF):
            print('Check fail')
            diff_coef = np.array([1.0,-1.0])
            diff_CSF = LC_CSFs([res_CSF_test,res_CSF],diff_coef)
            print_CSF(diff_CSF)
            sys.exit()
        assert len(res_CSF_test[0]) == len(res_CSF[0])

    return res_CSF

def identical_CSFs(CSF1,CSF2,tol=1e-6,debug=False):

    coefs = np.array([1.0,-1.0])

    resCSF = LC_CSFs([CSF1,CSF2],coefs)
    if debug:
        print('CSF1 in identical_CSFs:')
        print_CSF(CSF1)
        print('CSF2 in identical_CSFs:')
        print_CSF(CSF2)
        print(f'resultant CSF in identical_CSFs:')
        print_CSF(resCSF)

    norm_resCSF = overlap_CSFs(resCSF,resCSF)

   #if len(resCSF[0]) == 0:
    if norm_resCSF < tol:
        return True
    else:
        return False

def groupping_list_list_genmat(list_list_genmat,debug=False):
    """
    Decompose each genmat to normalized components, C, which satisfies C^3 = -C
    """

    import random
    if debug: print('\nIn groupping_list_list_genmat')

    list_list_decomp_genmat = []
    for list_genmat in list_list_genmat:
        list_decomp_genmat = []
        for genmat in list_genmat:
            ndim = genmat.shape[0]
            genmat_sq = genmat@genmat
            genmat_sq_dense = genmat_sq.toarray() if scipy.sparse.issparse(genmat_sq) else np.asarray(genmat_sq)
            eigval, eigvec = np.linalg.eigh(genmat_sq_dense)
            uniq_eigval = [eigval[0]]
            for ival in range(1,len(eigval)):
                if not np.isclose(eigval[ival],eigval[ival-1]):
                    uniq_eigval.append(eigval[ival])
           #print(f'unique eigvalues of genmat^2: {uniq_eigval}')
            list_eigvec = []
            list_start_end_ind = []
            list_prj_genmat = []
            list_unique_eigval = []
            for val in uniq_eigval:
                collist = np.where(abs(eigval - val) < 1.0e-6)[0]
               #print(f'collist for eigval {val}: {collist}')
                istart, iend = collist[0],collist[-1]
               #prjmat = eigvec[:,istart:iend+1]@eigvec[:,istart:iend+1].transpose()
               #prjmat = csr_matrix(prjmat)
                prj_genmat = eigvec[:,istart:iend+1].transpose()@genmat@eigvec[:,istart:iend+1]
                list_start_end_ind.append([istart,iend])
                list_eigvec.append(copy.deepcopy(eigvec[:,istart:iend+1]))
                list_unique_eigval.append(val)
                if not np.isclose(val,0.0): prj_genmat = prj_genmat / np.sqrt(-val)
                prj_genmat = csr_matrix(prj_genmat)
                list_prj_genmat.append(prj_genmat)
                prj_genmat_sq = prj_genmat@prj_genmat
               #print(f'prj_genmat_sq')
               #print_matrix(prj_genmat_sq.toarray())
                eigval_prj, _ = np.linalg.eigh(prj_genmat_sq.toarray())
                uniq_prjeigval = list(set(eigval_prj))
                if debug: print(f'unique eigvalues of prj_genmat^2: {uniq_prjeigval}')

            if debug:
                theta = random.uniform(0,1)
                tic = time.perf_counter()
                genmat_dense = genmat.toarray() if scipy.sparse.issparse(genmat) else np.asarray(genmat)
                exp_genmat = scipy.linalg.expm(theta*genmat_dense)
                toc = time.perf_counter()
                time_for_num_exp = toc - tic
               #print('\nNumerical exp(theta*G)')
               #print_matrix(exp_genmat)
                exp_genmat = csr_matrix(exp_genmat)

                tic = time.perf_counter()
                exp_genmat_anl = csr_matrix((ndim,ndim))
                for ival, unival in enumerate(list_unique_eigval):
                    [istart, iend] = list_start_end_ind[ival]
                    sub_eigvec = eigvec[:,istart:iend+1]
                    sub_eigvec = csr_matrix(sub_eigvec)
                    prj_genmat = list_prj_genmat[ival]
                    if np.isclose(unival,0.0):
                        assert np.isclose(scipy.sparse.linalg.norm(prj_genmat),0.0)
                        exp_genmat_anl += sub_eigvec @ sub_eigvec.transpose()
                    else:
                        ndim_sub = prj_genmat.shape[0]
                        sub_expgenmat = scipy.sparse.identity(ndim_sub,format="csr")
                        sub_expgenmat += prj_genmat*np.sin(np.sqrt(-unival)*theta)
                        sub_expgenmat -= prj_genmat@prj_genmat*(np.cos(np.sqrt(-unival)*theta)-1.0)
                        sub_expgenmat_numrc = scipy.linalg.expm(np.sqrt(-unival)*theta*prj_genmat.toarray())
                        exp_genmat_anl += sub_eigvec@sub_expgenmat@sub_eigvec.transpose()

               #print('\nAnalytical exp(theta*G)')
               #print_matrix(exp_genmat_anl.toarray())
                toc = time.perf_counter()
                time_for_anl_exp = toc - tic
                assert np.isclose(scipy.sparse.linalg.norm(exp_genmat_anl - exp_genmat),0.0)
                print(f'Time for numerical  expm: {time_for_num_exp}')
                print(f'Time for analytical expm: {time_for_anl_exp}')

            list_decomp_genmat.append([list_unique_eigval,list_prj_genmat,eigvec,list_start_end_ind])

        list_list_decomp_genmat.append(list_decomp_genmat)

    return list_list_decomp_genmat

def select_ia_pairs_for_CSF_E0_with_pairex_in_actmo(list_CSF,class_somo_ind,group_iapair,actmo_start,actmo_end,Enuc,obt,tbt,Ethrsh=1.e-3,debug=False):
    """
    Select ia pairs based on their possibility to decrease E0. The same class of CSFs
    have to share the same set of ia pairs.
    """

    import random
    if debug: print('\nIn select_ia_pairs_for_CSF_E0_with_pairex_in_actmo')
    nelec = len(np.where(list_CSF[0][0][0] == 1.0)[0])
    homo = nelec // 2 - 1
    print(f'homo: {homo}')
    list_pair_ex_in_actmo = []
    for ii in range(actmo_start,homo+1):
        for aa in range(homo+1,actmo_end+1):
            list_pair_ex_in_actmo.append([[ii,aa]])

    print(f'\nSelecting ia pair based on Energy lowering Thrshold {Ethrsh}\n')
    list_theta_0 = [0.0]*len(list_pair_ex_in_actmo)

    list_set_dmo = []
    for iCSF, CSF in enumerate(list_CSF):
        list_dmo = dmo_in_SD(CSF[0][0])
        set_dmo = set(list_dmo)
        print(f'DMOs of CSF{iCSF}: {set_dmo}')
        list_set_dmo.append(set_dmo)


    print(f'list_pair_ex_in_actmo: {list_pair_ex_in_actmo}')

    Hmat_CSF = construct_Hmat_CSFs(list_CSF,Enuc,obt,tbt)

    Hmat_CSF_sparse = csr_matrix(Hmat_CSF)
    E0_CSF, psi_GS_CSF = get_ground_state(Hmat_CSF)
    print(f'\nE0 of original CSF space: {E0_CSF}')

    list_list_ia_included = []
    list_list_ex_space = []
    list_list_CSF_ind_ex_space = []
    list_list_genmat = []
    for iCSF_class in range(len(class_somo_ind)):
        print(f'\nChecking ia pairs for Class {iCSF_class}')
        list_ia_one_class = []
        list_improve = []
        iCSF_start = class_somo_ind[iCSF_class][0]
        iCSF_end   = class_somo_ind[iCSF_class][1]
        list_ex_space = copy.deepcopy(list_CSF[iCSF_start:iCSF_end+1])
        for iCSF in range(iCSF_start,iCSF_end+1):
            print(f'Checking ia pairs for CSF{iCSF}')
            CSF = list_CSF[iCSF]
            l_adding_ia = False
            for ia_pair in group_iapair:
                UCSF,Uvec,list_pair_ex_space,Umat, _ = prepare_UCSF_for_one_CSF(CSF,[ia_pair],[0.0],False)
                if len(list_pair_ex_space) == 1: continue
                list_UCSF_space = list_CSF + list_pair_ex_space[1:]
               #if debug: print(ia_pair,len(list_UCSF_space),len(list_CSF))
                Hmat_UCSF = construct_Hmat_CSFs(list_UCSF_space,Enuc,obt,tbt)
                Hmat_UCSF_sparse = csr_matrix(Hmat_UCSF)
                E0_UCSF, psi_GS_UCSF = get_ground_state(Hmat_UCSF_sparse)
                lowering = E0_UCSF-E0_CSF
                if lowering > 1.0e-8: print(f'Strange, positive lowering: {E0_UCSF} vs {E0_CSF}')
                if abs(lowering) > Ethrsh:
                   if ia_pair not in list_ia_one_class:
                       list_ia_one_class.append(ia_pair)
                       list_improve.append(lowering)
                       l_adding_ia = True
                       l_exCSF_included = False
                       for item in list_pair_ex_space[1:]:
                           for CSFex in list_ex_space:
                               Selm = overlap_CSFs(item,CSFex)
                               if not np.isclose(Selm,0.0):
                                  #print('Strange! an EX CSF has been included before')
                                  #print(item)
                                   l_exCSF_included = True
                           if not l_exCSF_included: list_ex_space.append(item)
                   else:
                       ind_ia_pair = list_ia_one_class.index(ia_pair)
                       if lowering < list_improve[ind_ia_pair]:
                           print(f'Replacing list_improve: {ia_pair}, {ind_ia_pair},{list_improve[ind_ia_pair]},{lowering}')
                           list_improve[ind_ia_pair] = lowering

            if debug and l_adding_ia:
                print(f'Updated list_ia_one_class for Class: {iCSF_class}:')
                for ii in range(len(list_ia_one_class)):
                    print(list_ia_one_class[ii],list_improve[ii])

        if len(list_improve) == 0:
            list_list_ia_included.append(list_ia_one_class) #An empty list is appended
            list_list_ex_space.append(list_ex_space) #list_ex_space = the original CSFs and is appended
            list_list_CSF_ind_ex_space.append(list(range(len(list_ex_space)))) #Append a list of [0,1,2,...] as the order is the same
            list_list_genmat.append([])
        else:
            list_improve, list_ia_one_class = zip(*sorted(zip(list_improve,list_ia_one_class)))
            list_ia_one_class = list(list_ia_one_class)
            print(f'\nMaking ex CSF space of a whole class {iCSF_class}')
            list_random_theta = []
            for ii in range(len(list_ia_one_class)):
                list_random_theta.append(random.uniform(0,1))

            print(f'a list of randome thetas: {list_random_theta}')

            list_all_ia_one_class = list_pair_ex_in_actmo + list_ia_one_class
            list_all_theta_one_class = list_theta_0 + list_random_theta
            CSF = list_CSF[iCSF_start]
            UCSF, Uvec, list_pair_ex_space, Umat, list_genmat = prepare_UCSF_for_one_CSF(CSF,list_all_ia_one_class,list_all_theta_one_class,False)
           #list_genmat = list_genmat[len(list_pair_ex_in_actmo):]
            list_genmat = construct_list_genmat_from_occ(list_pair_ex_space,list_ia_one_class,debug=True)
            if iCSF_class == 3:
                for item in list_pair_ex_space:
                    print(f'list dmo: {dmo_in_SD(item[0][0])}')
                for igenmat,genmat in enumerate(list_genmat):
                    print(f'genmat {igenmat}:')
                    print_matrix(genmat.toarray())
            for iCSF in range(iCSF_start+1,iCSF_end+1):
                if list_set_dmo[iCSF] == list_set_dmo[iCSF_start]:
                    CSF = list_CSF[iCSF]
                    UCSF, Uvec, list_pair_ex_space_more, Umat, list_genmat_more = prepare_UCSF_for_one_CSF(CSF,list_all_ia_one_class,list_all_theta_one_class,False)
                    list_genmat_more = construct_list_genmat_from_occ(list_pair_ex_space_more,list_ia_one_class,debug=True)
                    if iCSF_class == 3:
                        for igenmat,genmat_more in enumerate(list_genmat_more):
                            print(f'genmat_more {igenmat}:')
                            print_matrix(genmat_more.toarray())
                    list_pair_ex_space += list_pair_ex_space_more
                   #Expand the genmats to accommodate the space expansion
                    [nrow_orig,ncol_orig] = list_genmat[0].shape
                    [nrow_new,ncol_new] = list_genmat_more[0].shape
                    print(nrow_orig,ncol_orig,nrow_new,ncol_new)
                    for ii in range(len(list_genmat)):
                        list_genmat[ii] = scipy.sparse.hstack([list_genmat[ii],csr_matrix((nrow_orig,ncol_new))])
                        list_genmat_more[ii] = scipy.sparse.hstack([csr_matrix((nrow_new,ncol_orig)),list_genmat_more[ii]])
                        list_genmat[ii] = scipy.sparse.vstack([list_genmat[ii],list_genmat_more[ii]])
                        print(f'Expanded genmat {ii}:')
                        print(list_genmat[ii])


            nCSF_in_ex_space = len(list_pair_ex_space)
            print(f'# of all CSFs in the ex space of Class {iCSF_class}: {nCSF_in_ex_space}')
            list_CSF_ind_in_ex_space = []
            for iCSF, CSF in enumerate(list_CSF[iCSF_start:iCSF_end+1]):
                l_CSF_found = False
                for iCSFex, CSFex  in enumerate(list_pair_ex_space):
                    Selm = overlap_CSFs(CSF, CSFex)
                    if np.isclose(abs(Selm),1.0):
                        l_CSF_found = True
                        list_CSF_ind_in_ex_space.append(iCSFex)
                        break

                if not l_CSF_found:
                    print(f'CSF{iCSF+iCSF_start} not found in EX space. Bombing out!')
                    sys.exit()

            print(f'list of indices of CSF in the exCSF space: {list_CSF_ind_in_ex_space}')

            list_list_ia_included.append(list_ia_one_class)
            list_list_ex_space.append(list_pair_ex_space)

            list_list_CSF_ind_ex_space.append(list_CSF_ind_in_ex_space)
            list_list_genmat.append(list_genmat)



        print(f'\nIncluded ia pairs for Class: {iCSF_class}:')
        for ii in range(len(list_ia_one_class)):
            print(list_ia_one_class[ii],list_improve[ii])


    return list_list_ex_space, list_list_ia_included, list_list_genmat, list_list_CSF_ind_ex_space

def select_ia_pairs_for_one_CSF(CSF,group_mp2_ia_pairs,group_mp2_ampld,Enuc,obt,tbt,Ethrsh=1.0e-3,debug=False):
    """
    select ia pairs that give U|CSF> for a read-in CSF
    """
    if debug: print('\nIn select_ia_pairs_for_one_CSF')

    list_ia_included = []
    list_improve = []
    list_ia_ampld_included = []
   #Ethrsh = 1.0e-5
    for ia in range(len(group_mp2_ia_pairs)):
        UCSF, Uvec, list_pair_ex_space, Umat, list_genmat =\
            prepare_UCSF_for_one_CSF(CSF,group_mp2_ia_pairs[ia:ia+1],[0.0],False)
        n_mp2_ampld_opt = 1
        E_CSF = Helm_between_CSFs(Enuc,obt,tbt,CSF,CSF)
        list_Ustate_opt, list_mp2_ampld_opt, improve, E_UCSF_opt = opt_U_for_GS_of_Hmat([list_pair_ex_space],[list_genmat],[[0.0]],Enuc,obt,tbt,[n_mp2_ampld_opt])
       #Improve is re-defined as the energy lowering compared to the unperturbed CSF
       #improve = E_UCSF_opt - E_CSF
        assert np.isclose(improve, E_UCSF_opt - E_CSF)
        if abs(improve) > Ethrsh:
            list_ia_included.append(group_mp2_ia_pairs[ia])
            list_improve.append(improve)
            list_ia_ampld_included.append(list_mp2_ampld_opt[0][0])

    if len(list_ia_included) == 0:
        print('No ia pairs are found. Bombing out!')
        sys.exit()
    list_improve, list_ia_included, list_ia_ampld_included = zip(*sorted(zip(list_improve,list_ia_included,list_ia_ampld_included)))
    list_ia_included = list(list_ia_included)
    list_ia_ampld_included = list(list_ia_ampld_included)
    if debug:
        for ii in range(len(list_improve)):
            print(list_improve[ii],list_ia_included[ii],list_ia_ampld_included[ii])

    UCSF, Uvec, list_pair_ex_space, Umat, list_genmat =\
      prepare_UCSF_for_one_CSF(CSF,list_ia_included,list_ia_ampld_included,False)

    return list_pair_ex_space, list_ia_included, list_ia_ampld_included, list_genmat

def opt_flexible_U_for_one_CSF(CSF,group_mp2_ia_pairs,group_mp2_ampld,Enuc,obt,tbt,Ethrsh=1.0e-3,debug=False):
    """
    optimize U|CSF> for a read-in CSF
    """
    if debug: print('\nIn opt_flexible_U_for_one_CSF')

    list_ia_included = []
    list_improve = []
    list_ia_ampld_included = []
   #Ethrsh = 1.0e-5
    for ia in range(len(group_mp2_ia_pairs)):
        UCSF, Uvec, list_pair_ex_space, Umat, list_genmat =\
            prepare_UCSF_for_one_CSF(CSF,group_mp2_ia_pairs[ia:ia+1],group_mp2_ampld[ia:ia+1],False)
       #print(group_mp2_ia_pairs[ia],len(list_pair_ex_space))
        n_mp2_ampld_opt = 1
       #list_Ustate_opt, list_mp2_ampld_opt, improve, _ = opt_U_for_GS_of_Hmat([list_pair_ex_space],[list_genmat],[group_mp2_ampld[ia:ia+1]],Enuc,obt,tbt,[n_mp2_ampld_opt])
        E_CSF = Helm_between_CSFs(Enuc,obt,tbt,CSF,CSF)
        list_Ustate_opt, list_mp2_ampld_opt, improve, E_UCSF_opt = opt_U_for_GS_of_Hmat([list_pair_ex_space],[list_genmat],[group_mp2_ampld[ia:ia+1]],Enuc,obt,tbt,[n_mp2_ampld_opt])
       #Improve is re-defined as the energy lowering compared to the unperturbed CSF
        improve = E_UCSF_opt - E_CSF
        list_ia_included.append(group_mp2_ia_pairs[ia])
        list_improve.append(improve)
        list_ia_ampld_included.append(list_mp2_ampld_opt[0][0])
       #print(f'list_mp2_ampld_opt: {list_mp2_ampld_opt}')
       #print(f'Energy lowring of {improve} for optimizing theta for {group_mp2_ia_pairs[ia:ia+1]}')


    list_l_remove = [True] * len(group_mp2_ia_pairs)
    l_next_loop = True
    scaled_Ethrsh = Ethrsh*10.0
    nround = 0
    print(f'Improvement in the initial one by one opt')
    for ii, pair in enumerate(group_mp2_ia_pairs):
        print(pair,list_improve[ii])
   #print(f'list_improve: {list_improve}')
    while(l_next_loop):
        nround += 1
       #print(f'nround: {nround}')
        scaled_Ethrsh *= 0.1
        for ia in range(len(group_mp2_ia_pairs)):
            if abs(list_improve[ia]) > scaled_Ethrsh:
                list_l_remove[ia] = False
        n_false = list_l_remove.count(False)
        if n_false != 0: l_next_loop = False
        if nround == 10:
            print(f'After {nround} rounds of reducing E threshold, still, no ia pairs are selected ')
            print(f'None of the pairs are selected. The original CSF is returned')
            return CSF, [CSF], [], [], []

    print(list_l_remove)
    for ii in range(len(group_mp2_ia_pairs)-1,-1,-1):
        if list_l_remove[ii]:
            del list_ia_included[ii]
            del list_improve[ii]
            del list_ia_ampld_included[ii]



#           list_ia_included.append(group_mp2_ia_pairs[ia])
#           list_improve.append(improve)
#           list_ia_ampld_included.append(list_mp2_ampld_opt[0][0])

   #print(f'list_ia_ampld_included: {list_ia_ampld_included}')
   #print(f'Included ia pairs before sorting')
   #for ii in range(len(list_ia_included)):
   #    print(list_ia_included[ii],list_improve[ii],list_ia_ampld_included[ii])
    list_improve, list_ia_included, list_ia_ampld_included = zip(*sorted(zip(list_improve,list_ia_included,list_ia_ampld_included)))
    list_ia_included = list(list_ia_included)
    list_ia_ampld_included = list(list_ia_ampld_included)
    print(f'The following {len(list_ia_included)} ia pairs pass the first round of screening:')
    for ii in range(len(list_ia_included)):
        print(list_ia_included[ii],list_improve[ii],list_ia_ampld_included[ii])

    UCSF, Uvec, list_pair_ex_space, Umat, list_genmat =\
        prepare_UCSF_for_one_CSF(CSF,list_ia_included,list_ia_ampld_included,False)

    n_mp2_ampld_opt=len(list_ia_ampld_included)
    list_Ustate_opt, list_mp2_ampld_opt, improve, _ = opt_U_for_GS_of_Hmat([list_pair_ex_space],[list_genmat],[list_ia_ampld_included],Enuc,obt,tbt,[n_mp2_ampld_opt])
    print(f'Improve in simultaneous opt of all included pairs: {improve}')
    E_CSF = Helm_between_CSFs(Enuc,obt,tbt,CSF,CSF)
    E_UCSF = Helm_between_CSFs(Enuc,obt,tbt,list_Ustate_opt[0],list_Ustate_opt[0])
    print(f'E_CSF: {E_CSF}, Energy of the opt state: {E_UCSF}, corr. E: {E_UCSF - E_CSF}')
    if debug:
        print(f'\nThe first round of simultaneous opt give the following thetas')
        for ii in range(len(list_mp2_ampld_opt[0])):
            print(list_ia_included[ii],list_mp2_ampld_opt[0][ii])

   #list_ia_train contains the whole train of ia rotation, including duplication of ia pairs.
   #list_ia_included does not contain the duplication. It only includes those in the first round opt.
    list_ia_train = copy.deepcopy(list_ia_included)
   #print(f'list_ia_train: {list_ia_train}')
   #2nd round, the last one is not included because it is trivial to have two identical generator back to back
    n_ia_pair_included = len(list_mp2_ampld_opt[0])
    n_mp2_ampld_opt = n_ia_pair_included + 1
    print(n_ia_pair_included,n_mp2_ampld_opt)

    l_next_round = True
    list_genmat_save = copy.deepcopy(list_genmat)
    list_theta_save = list_mp2_ampld_opt[0]
    list_ia_ind = []
    n_mp2_ampld_opt = len(list_theta_save)
    Ustate_opt = list_Ustate_opt[0]
    n_round = 0

    while(l_next_round): #l_next_round is never turned to False. The while loop is terminated by a "break" statement.
        list_include = []
        list_improve = []
        for ia in range(n_ia_pair_included):
            print(ia)
            list_genmat_2nd = copy.deepcopy(list_genmat_save) + [list_genmat[ia]]
            list_ia_ampld = copy.deepcopy(list_theta_save) + [0.0]
            list_Ustate_opt, list_mp2_ampld_opt_2nd, improve, _ = opt_U_for_GS_of_Hmat([list_pair_ex_space],[list_genmat_2nd],[list_ia_ampld],Enuc,obt,tbt,[n_mp2_ampld_opt+1],False)
            if abs(improve) >  Ethrsh:
                list_include.append([ia,list_mp2_ampld_opt_2nd[0][-1]])
                list_improve.append(improve)

        if len(list_include) == 0:
            print(f'No more ia rotation will be added. Opt converged after {n_round} rounds!')
            break

        n_round += 1
        if len(list_improve) != 0: list_improve, list_include = zip(*sorted(zip(list_improve,list_include)))
        list_genmat_2nd = copy.deepcopy(list_genmat_save)
        list_ia_ampld = copy.deepcopy(list_theta_save)
        n_mp2_ampld_opt += len(list_improve)
        print(f'n_mp2_ampld_opt = {n_mp2_ampld_opt}')
        for ii in range(len(list_include)):
            list_genmat_2nd += [list_genmat[list_include[ii][0]]]
           #list_ia_ampld = list_mp2_ampld_opt[0] + [list_include[ii][1]]
            list_ia_ampld += [0.0]
            list_ia_train.append(list_ia_included[list_include[ii][0]])
            print(list_include[ii],list_improve[ii],list_ia_train[-1])
        print(f'More rounds are needed')
        print(list_ia_ampld)
        list_Ustate_opt, list_mp2_ampld_opt_2nd, improve, _ = opt_U_for_GS_of_Hmat([list_pair_ex_space],[list_genmat_2nd],[list_ia_ampld],Enuc,obt,tbt,[n_mp2_ampld_opt],True)
        print(f'Improve in opt all thetas: {improve}')
        E_CSF = Helm_between_CSFs(Enuc,obt,tbt,CSF,CSF)
        E_UCSF = Helm_between_CSFs(Enuc,obt,tbt,list_Ustate_opt[0],list_Ustate_opt[0])
        print(f'Energy of the opt state: {E_UCSF}, corr. E: {E_UCSF - E_CSF}')
        print(list_mp2_ampld_opt_2nd[0])
        list_genmat_save = list_genmat_2nd
        list_theta_save = list_mp2_ampld_opt_2nd[0]
        Ustate_opt = list_Ustate_opt[0]

       #sys.exit()

    assert len(list_theta_save) == len(list_ia_train)
    assert len(list_theta_save) == len(list_genmat_save)
    if debug:
        print('\nThe following train of ia rotation applies to create UCSF:')
        for ii in range(len(list_ia_train)):
            print(list_theta_save[ii],list_ia_train[ii])

    return Ustate_opt, list_pair_ex_space, list_genmat_save, list_theta_save, list_ia_train

def prepare_UCSF_for_one_CSF_group(CSF, group_mp2_ia_pairs, group_mp2_ampld,
                                       extra_basis_CSF=None, nparal=1, debug=False):
    """
    Prepare U|CSF> for one CSF, appending extra_basis_CSF states to the
    excitation space without including them in the generator U.
    """
    if extra_basis_CSF is None:
        extra_basis_CSF = []

    if debug:
        print(f'\nIn prepare_UCSF_for_one_CSF_group')
        print(f'  {len(group_mp2_ia_pairs)} generator pairs, '
                  f'{len(extra_basis_CSF)} extra basis CSFs')

    # Flatten grouped pairs for make_pair_ex_space_group_manual
    ungroupped_ia_pair = [pair for pairs in group_mp2_ia_pairs for pair in pairs]

    list_pair_ex_space, list_genmat = make_pair_ex_space_group_manual(
        CSF, group_mp2_ia_pairs,extra_basis_CSF=extra_basis_CSF,
            nparal=nparal,debug=debug)

    if len(list_genmat) == 0:
        ndim_space = len(list_pair_ex_space)
        Umat = np.eye(ndim_space)
        Uvec = np.zeros(ndim_space)
        Uvec[0] = 1.0
    else:
        Umat, Uvec = make_and_apply_U_matrix(list_genmat, group_mp2_ampld)

    UCSF = make_UCSF_state(list_pair_ex_space, Uvec)

    return UCSF, Uvec, list_pair_ex_space, Umat, list_genmat

def prepare_UCSF_for_one_CSF(CSF,group_mp2_ia_pairs,group_mp2_ampld,nparal=1,debug=False):
    """
    Prepare U|CSF> for one CSF
    """

    if debug: print('\nIn prepare_UCSF_for_one_CSF')

    ungroupped_ia_pair = []
    for pairs in group_mp2_ia_pairs:
        for pair in pairs:
            ungroupped_ia_pair.append(pair)

    [UCSF], [Uvec], [list_pair_ex_space], [Umat], [list_genmat] = prepare_UCSF_generic([CSF],ungroupped_ia_pair,group_mp2_ia_pairs,group_mp2_ampld,nparal,debug)
    return UCSF, Uvec, list_pair_ex_space, Umat, list_genmat

#def prepare_UCSF_for_one_CSF_group(list_combined_CSFs, group_ia_pairs, group_ampld, nparal=1, debug=False):
    """
    Takes a pre-defined list of CSFs and calculates the generator matrices
    acting on that specific basis.
    """
    if debug: print(f'\nIn prepare_UCSF_for_CSF_group with {len(list_combined_CSFs)} CSFs')

    # 1. Flatten the ia pairs for the underlying logic if needed
    ungroupped_ia_pair = [pair for pairs in group_ia_pairs for pair in pairs]

    list_pair_ex_space, list_genmat = make_pair_ex_space_manual_for_group(
        list_combined_CSFs, group_ia_pairs, nparal=nparal, debug=debug
    )

    # 3. Apply the unitary rotation to the reference state (index 0)
    if len(list_genmat) == 0:
        ndim_space = len(list_pair_ex_space)
        Umat = np.eye(ndim_space)
        Uvec = np.zeros(ndim_space)
        Uvec[0] = 1.0
    else:
        # This builds the U matrix and applies it to [1, 0, 0...]
        Umat, Uvec = make_and_apply_U_matrix(list_genmat, group_ampld)

    # 4. Construct the final state
    UCSF = make_UCSF_state(list_pair_ex_space, Uvec)

    return UCSF, Uvec, list_pair_ex_space, Umat, list_genmat

def prepare_UCSF_one_CSF_one_ia_manual(CSF,list_one_pair,theta,nparal=1,debug=False):

    tic = time.perf_counter()
    if debug: print(f'In prepare_UCSF_one_CSF_one_ia_manual')

    ia_pair = list_one_pair[0]
    T_op = get_Tiiaa_00(ia_pair[0],ia_pair[1])
    PT_iiaa = -T_op*T_op
    PT_iiaa = normal_ordered(PT_iiaa)
    PT_iiaa.compress()

   #PCSF_test =  op_action_tz_CSF(PT_iiaa,CSF)
   #onlist_post, idxlist_post, coefs_post = op_action_tz_remove_0coef(PT_iiaa,CSF[0],CSF[1],CSF[2])
   #PCSF = [onlist_post, idxlist_post, coefs_post]
   #assert identical_CSFs(PCSF_test,PCSF)
   #PCSF = op_action_tz_CSF(PT_iiaa,CSF)
    tic_PCSF = time.perf_counter()
    PCSF = op_action_tz_CSF_in_chunks(PT_iiaa,CSF,nparal)
    toc_PCSF = time.perf_counter()
    print(f'Time to prepare PCSF: {toc_PCSF - tic_PCSF}')
    coef_p = np.cos(theta) - 1.0

   #TCSF_test = op_action_tz_CSF(T_op,CSF)
   #onlist_post, idxlist_post, coefs_post = op_action_tz_remove_0coef(T_op,CSF[0],CSF[1],CSF[2])
   #TCSF = [onlist_post, idxlist_post, coefs_post]
   #assert identical_CSFs(TCSF_test,TCSF)
   #TCSF = op_action_tz_CSF(T_op,CSF)
    tic_TCSF = time.perf_counter()
    TCSF = op_action_tz_CSF_in_chunks(T_op,CSF,nparal)
    toc_TCSF = time.perf_counter()
    print(f'Time to prepare TCSF: {toc_TCSF - tic_TCSF}')
    coef_t = np.sin(theta)
    coefs = np.array([1.0,coef_p,coef_t])
    tic_LC = time.perf_counter()
    UCSF = LC_CSFs([CSF,PCSF,TCSF],coefs,False)
    toc_LC = time.perf_counter()
    print(f'Time for LC_CSFs to combine three parts: {toc_LC - tic_LC}')

    toc = time.perf_counter()
    print(f'Time for prepare_UCSF_one_CSF_one_ia_manual: {toc - tic}')

    return UCSF

def prepare_UCSF_one_CSF_Tiiaa_Tiibb_manual(CSF,ia_pair,ib_pair,theta,debug=False):
    """
    T = 1/sqrt(2)(Tiiaa + Tiibb)
    U = exp(theta*T)
    return |UCSF> = U|CSF>
    """

    if debug: print(f'In prepare_UCSF_one_CSF_Tiiaa_Tiibb_manual')


   #assert ia_pair[0] == ib_pair[0] and ia_pair[1] != ib_pair[1]

    if ia_pair[0] == ib_pair[0] and ia_pair[1] != ib_pair[1]:
        orb_nondegen = ia_pair[0]
        orb_degen1 = ia_pair[1]
        orb_degen2 = ib_pair[1]
        l_change_sign = False
    elif ia_pair[0] != ib_pair[0] and ia_pair[1] == ib_pair[1]:
        orb_nondegen = ia_pair[1]
        orb_degen1 = ia_pair[0]
        orb_degen2 = ib_pair[0]
        l_change_sign = True
    else:
        print(f'The not [[i,a],[i,b]] or [[i,a],[j,a]] pairs detected: {ia_pair,ib_pair}')
        sys.exit()

   #Top = get_sqrt_half_Tiiaa_00_plus_Tiibb_00(ia_pair[0],ia_pair[1],ib_pair[1])
    Top = get_sqrt_half_Tiiaa_00_plus_Tiibb_00(orb_nondegen,orb_degen1,orb_degen2)
    if l_change_sign: Top = -Top

   #Proj, Proj2e, Proj4e, Proj3eup, Proj3edn = proj_Tiiaa_plus_Tiibb(ia_pair[0],ia_pair[1],ib_pair[1])
    Proj, Proj2e, Proj4e, Proj3eup, Proj3edn = proj_Tiiaa_plus_Tiibb(orb_nondegen,orb_degen1,orb_degen2)

    Proj24 = normal_ordered(Proj2e + Proj4e)
    Proj24.compress()
    Proj3  = normal_ordered(Proj3eup+Proj3edn)
    Proj3.compress()

    P2CSF = op_action_tz_CSF(Proj2e,CSF)
    norm_P2CSF = overlap_CSFs(P2CSF,P2CSF)
   #if not np.isclose(norm_P2CSF,0.0):
   #    print(f'P2CSF != 0, {norm_P2CSF}')
    P4CSF = op_action_tz_CSF(Proj4e,CSF)
    norm_P4CSF = overlap_CSFs(P4CSF,P4CSF)
   #if not np.isclose(norm_P4CSF,0.0):
   #    print(f'P4CSF != 0, {norm_P4CSF}')
    P3CSF = op_action_tz_CSF(Proj3,CSF)
    norm_P3CSF = overlap_CSFs(P3CSF,P3CSF)
   #if not np.isclose(norm_P3CSF,0.0):
   #    print(f'P3CSF != 0, {norm_P3CSF}')


    lP2  = not np.isclose(norm_P2CSF,0.0)
    lP4  = not np.isclose(norm_P4CSF,0.0)
    lP3  = not np.isclose(norm_P3CSF,0.0)
    lP24 = lP2 or lP4

    if not lP24 and not lP3: return CSF
    if lP24 and lP3:
        print(f'The input CSF has nonzero components in both Proj24 and Proj3.')
        print(f'This is impossible')
        print('CSF:')
        print_CSF(CSF)
        sys.exit()

    if lP24:
        PCSF = LC_CSFs([P2CSF,P4CSF],np.ones(2),False)
        TPCSF = op_action_tz_CSF(Top,PCSF)
        coef_p = np.cos(theta)-1.0
        coef_t = np.sin(theta)
    elif lP3:
        PCSF = P3CSF
        TPCSF = op_action_tz_CSF(Top,P3CSF)
        sqrt2 = np.sqrt(2.0)
        coef_p = np.cos(theta/sqrt2)-1.0
        coef_t = sqrt2*np.sin(theta/sqrt2)

   #[PCSF,TPCSF]
    cvec = np.array([1.0,coef_p,coef_t])
   #print(f'PCSF:')
   #print_CSF(PCSF)
   #print(f'TCSF:')
   #print_CSF(TPCSF)
   #print(f'cvec: {cvec}')
    UCSF = LC_CSFs([CSF,PCSF,TPCSF],cvec,False)

    Selm_CSF = overlap_CSFs(CSF,CSF)
    Selm_UCSF = overlap_CSFs(UCSF,UCSF)
    if not np.isclose(Selm_CSF,Selm_UCSF):
        print(f'Selm_CSF: {Selm_CSF}, vs Selm_UCSF: {Selm_UCSF}')
        print('CSF:')
        print_CSF(CSF)
        print(f'lP2,lP4,lP24,lP3: {lP2,lP4,lP24,lP3}')
        print('PCSF')
        print_CSF(PCSF)
        print('TPCSF')
        print_CSF(TPCSF)
        sys.exit()

    return UCSF


    Uop = FermionOperator.identity() - Proj24 - Proj3
    Uop += (np.cos(theta) + np.sin(theta)*Top)*Proj24
    Uop += (np.cos(theta*np.sqrt(0.5)) + np.sqrt(2.0)*np.sin(theta*np.sqrt(0.5))*Top)*Proj3

    Uop = normal_ordered(Uop)
    Uop.compress()

    if debug:
        shall_be_one = normal_ordered(hermitian_conjugated(Uop)*Uop)
        shall_be_one.compress()

        assert FermionOperator.isclose(shall_be_one,FermionOperator.identity(),tol=1e-6)

   #onlist_post, idxlist_post, coefs_post = op_action_tz_remove_0coef(Uop,CSF[0],CSF[1],CSF[2])

    UCSF = op_action_tz_CSF(Uop,CSF)

    Selm_CSF = overlap_CSFs(CSF,CSF)
    Selm_UCSF = overlap_CSFs(UCSF,UCSF)

    assert check_zero(Top*Proj3*Top*Proj3+0.5*Proj3)

    if not np.isclose(Selm_CSF,Selm_UCSF):
        print(f'Non-unitary detected, {Selm_CSF,Selm_UCSF}')
        print(f'CSF:')
        print_CSF(CSF)
        print(f'UCSF:')
        print_CSF(UCSF)
        sys.exit()

    if debug:
        print(f'UCSF with 1/sqrt(2)(Tiiaa+Tiibb)')
        print_CSF(UCSF)

    return UCSF

def prepare_UCSF_one_CSF_manual(CSF,list_ia_pair,list_theta,nparal=1,debug=False):
    """
    Prepare UCSF by acting operators explicitly. list_ia_pair = [[[h,l]],[[h-1,l]], ...]
    The ia pairs have been groupped.
    """

   #print(f'In prepare_UCSF_one_CSF_manual, list_ia_pair: {list_ia_pair}')
    UCSF = copy.deepcopy(CSF)
    if len(list_ia_pair) == 0: return UCSF
    for ii, ia_pair in enumerate(list_ia_pair):
        tic = time.perf_counter()
        print(f'ia_pair {ii} in total {len(list_ia_pair)}')
        print(f'dimension of UCSF: {len(UCSF[0])}')
        theta = list_theta[ii]
        if len(ia_pair) == 1: #Only for [[i,a]] entry
            UCSF = prepare_UCSF_one_CSF_one_ia_manual(UCSF,ia_pair,theta,nparal)
        if len(ia_pair) == 2:
            if not bool(set(ia_pair[0]) & set(ia_pair[1])):
               #[[5,7],[6,8]]-type group
               #print(f'The two groupped pairs do not share indices')
               #print(ia_pair)
               #sys.exit()
                UCSF = prepare_UCSF_one_CSF_one_ia_manual(UCSF,[ia_pair[0]],theta,nparal)
                UCSF = prepare_UCSF_one_CSF_one_ia_manual(UCSF,[ia_pair[1]],theta,nparal)
            else:
               #[[5,9],[6,9]]- and [[4,7],[4,8]]-type groups
                UCSF = prepare_UCSF_one_CSF_Tiiaa_Tiibb_manual(UCSF,ia_pair[0],ia_pair[1],theta)
        toc = time.perf_counter()
        print(f'Time for this ia_pair: {toc - tic}')

    if debug:
        print(f'resultant UCSF:')
        print(UCSF)

    return UCSF

def E_UCSF_manual(CSF,list_ia_pair,list_theta,Enuc,obt,tbt,debug=False):
    """
    Energy of a UCSF
    """

    UCSF = prepare_UCSF_one_CSF_manual(CSF,list_ia_pair,list_theta)
    return Helm_between_CSFs(Enuc,obt,tbt,UCSF,UCSF), UCSF

def opt_E_UCSF_manual(CSF,list_ia_pair,list_theta_init,Enuc,obt,tbt,ldisp=False,debug=False):

    assert len(list_ia_pair) == len(list_theta_init)

    x0 = np.array(list_theta_init)
    def cost(x):
        return E_UCSF_manual(CSF,list_ia_pair,x,Enuc,obt,tbt)[0]

    options = {
    'maxiter' : 10000,
    'disp'    : ldisp,
    'gtol'    : 1.0e-4
    }

    sol = minimize(cost, x0, method='BFGS',options=options)

    vec_theta_opt = sol.x
   #print(f'Energy lowering: {cost(vec_theta_opt)-cost(x0)}')
    E_lower = cost(vec_theta_opt)-cost(x0)

    UCSF_opt = prepare_UCSF_one_CSF_manual(CSF,list_ia_pair,vec_theta_opt)

    return UCSF_opt,vec_theta_opt, E_lower

#def prepare_UCSFiiaa(list_bound,list_hf_pair_ex,Umat,debug=False):
#    """
#    Prepare U|CSFiiaa> states, given the read-in dmo-to-vmo excited space of a HF reference
#    list_bound = [i_low,i_up,a_low,a_up], the lower and upper bounds of hole and particle orbitals
#    """
#
#    if debug: print('\nIn prepare_UCSFiiaa')
#
#    ndim = len(list_hf_pair_ex)
#    [i_low, i_up, a_low, a_up] = list_bound
#    Uiiaa_basis = []
#    Uiiaa_CIS_list = []
#    for i in range(i_up,i_low-1,-1):
#        for a in range(a_low,a_up+1):
#            lfound = False
#            for istate, state in enumerate(list_hf_pair_ex):
#                onlist = state[0]
#                coefs  = state[2]
#                occ_i = nocc_spatial_orb_LCSD(onlist,coefs,i)
#                occ_a = nocc_spatial_orb_LCSD(onlist,coefs,a)
#                if np.isclose(occ_i,0.0) and np.isclose(occ_a,2.0):
#                    lfound = True
#                    break
#           #It is normal not to find the doubly-excited states. Such an excitation may correspond to small mp2 amplitude
#           #and got screend off in the very beginning.
#           #if not lfound:
#           #    print(f'The {i,a} pair excited state is not found in the list of pair excited states from HF reference')
#           #    print('Bombing out')
#           #    sys.exit()
#            if not lfound: continue
#            if debug: print(i,a,istate,state)
#            Uiiaa_vec = np.zeros([ndim])
#            Uiiaa_vec[istate] = 1.0
#            Uiiaa_vec = Umat@Uiiaa_vec
#            Uiiaa_state = make_UCSF_state(list_hf_pair_ex,Uiiaa_vec)
#            Uiiaa_basis.append(Uiiaa_state)
#            Uiiaa_CIS_list.append([[i,a],[i,a]])
#
#
#    return Uiiaa_basis, Uiiaa_CIS_list

def prepare_UCSF_pairs_in_subspace(list_bound,list_hf_pair_ex,Umat,n_pair_desired=0,debug=False):
    """
    Prepare U|CSF_I> states with I of all possible pair combination excitations within a space
    of a HF reference.
    list_bound = [i_low,i_up,a_low,a_up]. Orbitals lower than i_low should all have full occupations, and
    those higher than a_up should all have zero occupations
    """

    if debug:
        print('\nIn prepare_UCSF_pairs_in_subspace')
        print('States in hf pair ex space:')
        for item in list_hf_pair_ex:
            print(item)


    lbomb = False
    if len(list_bound) != 4: lbomb = True
    if debug: print(f'list_bound: {list_bound}')
    for i, orb in enumerate(list_bound[:-2]):
        if orb > list_bound[i+1]: lbomb = True

    if lbomb:
        print(f'\nInappropriate list_bound: {list_bound}')
        print('Bombing out')
        sys.exit()

    ndim = len(list_hf_pair_ex)
    [i_low, i_up, a_low, a_up] = list_bound


    n_spinorb = len(list_hf_pair_ex[0][0][0])
    n_spatialorb = n_spinorb // 2
   #print(list_hf_pair_ex[0][0][0])
   #nel = np.sum(list_hf_pair_ex[0][0][0])
   #hdmo = nel // 2 - 1
    if debug: print(f'n_spatialorb: {n_spatialorb}')

    Upairs_basis = []
    Upairs_CIS_list = []
    Upairs_vec_list = []

    for istate, state in enumerate(list_hf_pair_ex):
        if istate == 0: continue #Skip the first reference state
        onlist = state[0]
        coefs  = state[2]
        l_exclude = False
        for p in range(i_low):
            if not np.isclose(nocc_spatial_orb_LCSD(onlist,coefs,p),2.0): l_exclude = True
        for p in range(a_up+1,n_spatialorb):
            if not np.isclose(nocc_spatial_orb_LCSD(onlist,coefs,p),0.0): l_exclude = True
        if not l_exclude:
            i_list = []
            a_list = []
            for p in range(i_low,a_up+1):
                occ_p = nocc_spatial_orb_LCSD(onlist,coefs,p)
                if np.isclose(occ_p,0.0) and p <= i_up: i_list.append(p)
                if np.isclose(occ_p,2.0) and p >= a_low: a_list.append(p)

            if len(i_list) != len(a_list):
                print(f'\nInconsistent # of hole pairs and particle pairs in subspace {i_low,a_up}')
                print(f'# of hole pairs: {len(i_list)} in {i_list}')
                print(f'# of part pairs: {len(a_list)} in {a_list}')
                print('Bombing out!')
                sys.exit()

            n_pair_ex = len(i_list)
            if n_pair_desired != 0 and n_pair_ex != n_pair_desired: l_exclude = True


        if debug:
            if l_exclude:
               print(f'Excluded: {state}')
            else:
               print(f'Included: {state}')

        if not l_exclude:
            Upairs_vec = np.zeros([ndim])
            Upairs_vec[istate] = 1.0
            Upairs_vec = Umat@Upairs_vec
            Upairs_state = make_UCSF_state(list_hf_pair_ex,Upairs_vec)
            Upairs_basis.append(Upairs_state)

           #i_list = []
           #a_list = []
           #for p in range(i_low,a_up+1):
           #    occ_p = nocc_spatial_orb_LCSD(onlist,coefs,p)
           #    if np.isclose(occ_p,0.0) and p <= i_up: i_list.append(p)
           #    if np.isclose(occ_p,2.0) and p >= a_low: a_list.append(p)

           #if len(i_list) != len(a_list):
           #    print(f'\nInconsistent # of hole pairs and particle pairs in subspace {i_low,a_up}')
           #    print(f'# of hole pairs: {len(i_list)} in {i_list}')
           #    print(f'# of part pairs: {len(a_list)} in {a_list}')
           #    print('Bombing out!')
           #    sys.exit()

            pair_list = []
            for ii in range(len(i_list)):
                pair_list.append([i_list[ii],a_list[ii]])
                pair_list.append([i_list[ii],a_list[ii]])

            Upairs_CIS_list.append(pair_list)
            Upairs_vec_list.append(Upairs_vec)

    if debug:
        for ii, state in enumerate(Upairs_basis):
            print(Upairs_CIS_list[ii])

    return Upairs_basis,Upairs_CIS_list, Upairs_vec_list


def nocc_spatial_orb_SD(onvec,p_spatialmo):
    """
    Return occupation number of a spatial orbital in a SD with ON vector onvec
    """

    pa, pb = p_spatialmo*2, p_spatialmo*2+1

    np = onvec[pa] + onvec[pb]

    return np

def nocc_spatial_orb_LCSD(onlist,coefvec,p_spatialmo):
    """
    Return average occupation number of a spatial orbital in a linear combination of SDs
    """

    np = 0.0
    for ii, on in enumerate(onlist):
        coef = coefvec[ii]
        np += coef*coef*nocc_spatial_orb_SD(on,p_spatialmo)

    return np

def get_SOMO_in_CSF(CSF):
    """
    Return a list of spatial orbitals whose occupancies are 1
    """

    onlist = CSF[0]
    coefvec = CSF[2]
    n_spinmo = len(onlist[0])
    n_spatialmo = n_spinmo // 2
    list_somo = []
    for imo in range(n_spatialmo):
        occ_imo = nocc_spatial_orb_LCSD(onlist,coefvec,imo)
        if np.isclose(occ_imo,1.0): list_somo.append(imo)

    return list_somo

def dmo_in_SD(onvec):
    """
    Return a list of doubly occupied orbitals of an ON vector
    """

    list_dmo = []
    n_spatial = len(onvec) // 2
    for ii in range(n_spatial):
        occ = onvec[2*ii] + onvec[2*ii+1]
        if np.isclose(occ,2.0): list_dmo.append(ii)

    return list_dmo

def calc_Hmat_diagonal_block(list_pair_ex_space,Enuc,obt_phys,tbt_phys,debug=False):
    """
    Given a set of state generated by multi-pair excitations of the reference (0th state), calculate their
    Hamiltonian matrix elements
    """

    if debug: print('\nIn Hmat_diagonal_block')

    ndim = len(list_pair_ex_space)
    Hmat = np.zeros([ndim,ndim])

    for ii, state in enumerate(list_pair_ex_space):
        onlist_i = state[0]
        coefs_i  = state[2]
        for jj in range(ii,ndim):
            onlist_j = list_pair_ex_space[jj][0]
            coefs_j  = list_pair_ex_space[jj][2]
            Helm = Helm_between_LCSDs(Enuc,obt_phys,tbt_phys,onlist_i,coefs_i,onlist_j,coefs_j)
            Hmat[ii,jj] = Helm
            Hmat[jj,ii] = Helm

    return Hmat

def read_fcidump(filename,debug=False):
    """
    Read Hamiltonian from fcidump file. It is taken from Josh Cantin
    """

    import pyscf.tools.fcidump

    fcidump_dic = pyscf.tools.fcidump.read(filename, molpro_orbsym=True, verbose=True)
    num_orbitals = int(fcidump_dic['NORB'])
    tbt_full = pyscf.ao2mo.restore("s1",fcidump_dic['H2'],num_orbitals)
    tbt_phys_spatial = np.transpose(tbt_full, [0,3,1,2])
   #tbt_phys_spatial = tbt_full
    obt_spatial = fcidump_dic['H1']
    Ecore = fcidump_dic['ECORE']
    if debug:
        print(f'Dimensions of obt_spatial: {obt_spatial.shape}')
        print(f'Dimensions of tbt_phys_spatial: {tbt_phys_spatial.shape}')

    return Ecore, obt_spatial, tbt_phys_spatial
   #stupid line

def explicit_calc_matelm(state_bra,state_ket,op):
    """
    Calculate matrix element explicitly using braket_tz
    """

    onlist_ket = state_ket[0]
    coefs_ket = state_ket[2]
    onlist_bra = state_bra[0]
    coefs_bra = state_bra[2]

    elm = 0.0
    for ibra, onl in enumerate(onlist_bra):
        coef_l = coefs_bra[ibra]
        for jket, onr in enumerate(onlist_ket):
            coef_r = coefs_ket[jket]
            elm += coef_l*coef_r*braket_tz(onl,onr,op)

    return elm

def SD_CSF_CoupCoef(S_by2,M_by2,tN_by2,sigma_by2):
    """
    The Clebsch-Gordan coefficients in Eq. 2.6.5 and 2.6.6 of the Helgaker book.
    All inputs are integers and are the variables in the equations multiplied by 2.
    """

    if sigma_by2 != 1 and sigma_by2 != -1:
        print(f'sigma_by2 {sigma_by2}, is neither 1 or -1')
        print('Bombing out')
        sys.exit()

    S = float(S_by2) / 2.0
    M = float(M_by2) / 2.0
    sigma = float(sigma_by2) / 2.0

    if tN_by2 == 1:
        CoupCoef = S + 2.0*sigma*M
        CoupCoef /= 2.0*S
        CoupCoef = np.sqrt(CoupCoef)
    elif tN_by2 == -1:
        CoupCoef = S + 1.0 - 2.0*sigma*M
        CoupCoef /= 2.0*(S+1.0)
        CoupCoef = np.sqrt(CoupCoef)
        CoupCoef *= -2.0*sigma
    else:
        print(f'Unrecognized tN_by2 {tN_by2}, which shall only be +1 and -1.')
        print('Bombing out!')
        sys.exit()

    return CoupCoef

def generate_CASCI_space(Norb_total, Nel_total, Nactorb, Nactel, S_by2, list_seniority=[], list_mo_exclud=[],
                             l_pair_ex=True, debug=False):
    """
    Generate all CSFs in a CAS space with total spin S. The integral SX2 is read-in
    as S_by2. S_by2 also gives the minimum number of singly occupied orbitals
    """

    from itertools import combinations

    import copy

    if debug: print('\nIn generate_CASCI_space')

    N_SOMO_min = S_by2
    if (Nactel - S_by2) % 2 != 0:
        print(f'The number of active electrons {Nactel} is inconsistent with SX2 {S_by2}')
        print('Bombing out!')
        sys.exit()

    N_SOMO_max = min(2*Nactorb - Nactel, Nactel)

    Ninactorb = (Nel_total - Nactel) // 2
    if debug:
        print(f'N_SOMO_min {N_SOMO_min}, N_SOMO_max {N_SOMO_max}')
        print(f'# of inactive orbitals: {Ninactorb}')

    onvec_core = np.zeros([2*Norb_total])
    onvec_core[0:2*Ninactorb] = 1.0
    if debug: print(f'onvec_core: {onvec_core}')

    list_actmo = []
    for orb in range(Ninactorb,Ninactorb+Nactorb):
        if orb in list_mo_exclud: continue
        list_actmo.append(orb)

    array_of_1_one = np.ones(1)
    list_CSF = []
    list_SOMO_DMO = []
    if len(list_seniority) != 0:
        lbomb = False
        if min(list_seniority) < N_SOMO_min: lbomb = True
        if max(list_seniority) > Nactel:
            lbomb = True
            print('seniority > # of active electrons')
        for seniority in list_seniority:
            if (seniority + N_SOMO_min) % 2 != 0: lbomb = True

        if lbomb:
            print(f'Incompatible list_seniority: {list_seniority} and N_SOMO_min: {N_SOMO_min}')
            print('Bombing out!')
            sys.exit()

    else:
        for N_SOMO in range(N_SOMO_min, N_SOMO_max+1,2):
            list_seniority.append(N_SOMO)

    print(f'list_seniority: {list_seniority}')
   #for N_SOMO in range(N_SOMO_min, N_SOMO_max+1,2):
    for N_SOMO in list_seniority:

        N_alpha = (N_SOMO + S_by2) // 2
        N_beta  = (N_SOMO - S_by2) // 2

        l_opensh = False
        if N_SOMO != 0:
            l_opensh = True
            list_CSF_prototype = geneological_SD_CSF(N_alpha, N_beta,False)
            if debug:
                print(f'\n# of SOMOs: {N_SOMO}')
                print(f'Dimension of list_CSF_prototype {len(list_CSF_prototype)}')
                print('Compare it with Fig 2.1 of Helgaker book')

        Nel_pair = Nactel - N_SOMO
        if Nel_pair % 2 != 0:
             print(f'Nactel, {Nactel}, N_SOMO, {N_SOMO}, Nel_pair, {Nel_pair} not even')
             print(f'Bombing out')
             sys.exit()
        Ndmo = Nel_pair // 2
        list_SOMO_comb = list(combinations(list_actmo,N_SOMO))
       #print(list_SOMO)
        for SOMO_comb in list_SOMO_comb:

            list_CSF_SOMO = []
            if l_opensh:
                for CSF_proto in list_CSF_prototype:
               #    print(CSF_proto)
                    onlist = []
                    for iSD, SD in enumerate(CSF_proto[0]):
                        onvec = copy.deepcopy(onvec_core)
                        for iSOMO, SOMO in enumerate(SOMO_comb):
                            onvec[2*SOMO:2*SOMO+2] = SD[2*iSOMO:2*iSOMO+2]

                       #if debug:
                       #    print(SOMO_comb,SD)
                       #    print(onvec)
                        onlist.append(onvec)

                    list_CSF_SOMO.append(onlist)


                assert len(list_CSF_prototype) == len(list_CSF_SOMO)
               #print(list_CSF_SOMO)

            list_left_orb = copy.deepcopy(list_actmo)
            for SOMO in SOMO_comb:
                list_left_orb.remove(SOMO)
           #print(f'SOMOs: {SOMO_comb}, left over orbitals: {list_left_orb}')
            list_DMO_comb = list(combinations(list_left_orb,Ndmo))
            dmo_sum_1st = np.sum(list_DMO_comb[0])
            for DMO_comb in list_DMO_comb:
                dmo_sum = np.sum(DMO_comb)
                if dmo_sum < dmo_sum_1st:
                    print('The 0th DOM_comb does not give the smallest summation. Watching out!!!')
                    print(list_DMO_comb[0],DMO_comb)
                    sys.exit()
                list_VMO = copy.deepcopy(list_left_orb)
                for MO in DMO_comb:
                    list_VMO.remove(MO)

                if debug: print(f'SOMOs: {SOMO_comb}, DMOs: {DMO_comb}, VMOs: {list_VMO}')

                if l_opensh:
                    for iCSF_SOMO, CSF_SOMO in enumerate(list_CSF_SOMO):
                        onlist = []
                        onidx_list = []
                       #print(CSF_SOMO)
                        for on in CSF_SOMO:
                            onvec = copy.deepcopy(on)
                           #print(f'onvec: {onvec}')
                            for dmo in DMO_comb:
                                onvec[2*dmo:2*dmo+2] = 1.0
                           #print(onvec)
                            onlist.append(onvec)
                            onidx_list.append(get_on_idx(onvec))
                        list_CSF.append([onlist,onidx_list,list_CSF_prototype[iCSF_SOMO][2]])
                        list_SOMO_DMO.append([SOMO_comb,DMO_comb])
                else:
                    onvec = copy.deepcopy(onvec_core)
                    for dmo in DMO_comb:
                        onvec[2*dmo:2*dmo+2] = 1.0

                    onidx = get_on_idx(onvec)
                    list_CSF.append([[onvec],[onidx],array_of_1_one])
                    list_SOMO_DMO.append([SOMO_comb,DMO_comb])

               #Only the first in the combinations of dmos is considered if no pair excitations are considered
                if not l_pair_ex:
                    break

   #print(list_CSF[0])
   #print(list_CSF[-1])
    n_CSF = len(list_CSF)
    print(f'Total # of CSF states: {n_CSF}')

   #Check whether the CSFs are all orthonormal and have desired spin eigenvalues
    if debug: check_spin_adapted_CSF(list_CSF,S_by2)

    return list_CSF, list_SOMO_DMO


def geneological_SD_CSF(N_alpha, N_beta,debug=False,l_get_op=False):
    """
    Geneological coupling between SDs and CSFs with specific numbers of alpha and
    beta electrons
    """

    if debug: print(f'\nIn geneological_SD_CSF, N_alpha = {N_alpha}, N_beta = {N_beta}')


    N_open = N_alpha + N_beta
    list_SOMO = []
    for i in range(N_open):
        list_SOMO.append(i)

    from itertools import combinations

    comb_alpha_orb = list(combinations(list_SOMO,N_alpha))
    if debug: print(comb_alpha_orb)

    list_pvec = []
    list_tvec = []
    for item in comb_alpha_orb:
        pvec = np.full(N_open,-1)
        for orb in item:
            pvec[orb] = 1

       #print(pvec)
        list_pvec.append(pvec)
        l_tvec = True
        for orb in range(N_open):
            if np.sum(pvec[:orb]) < 0:
                l_tvec = False
                break

        if l_tvec: list_tvec.append(pvec)


    if debug:
        print('\nlist_pvec:')
        for pvec in list_pvec:
            print(pvec)
        print('\nlist_tvec:')
        for tvec in list_tvec:
            print(tvec)

    list_Pvec = []
    for item in list_pvec:
       #print(item)
        Pvec = np.full(N_open,0)
        for i in range(len(item)):
           #print(i,np.sum(item[:i+1]))
            Pvec[i] = np.sum(item[:i+1])
       #print(Pvec)
        list_Pvec.append(Pvec)

    list_Tvec = []
    for item in list_tvec:
       #print(item)
        Tvec = np.full(N_open,0)
        for i in range(len(item)):
           #print(i,np.sum(item[:i+1]))
            Tvec[i] = np.sum(item[:i+1])
       #print(Tvec)
        list_Tvec.append(Tvec)

    if debug:
        print('\nlist_Pvec:')
        for Pvec in list_Pvec:
            print(Pvec)
        print('\nlist_Tvec:')
        for Tvec in list_Tvec:
            print(Tvec)

    list_CSF = []
    list_C_op = []

    #loop over all CSFs
    for iCSF in range(len(list_Tvec)):
        Tvec = list_Tvec[iCSF]
        tvec = list_tvec[iCSF]
        list_Pvec_include = []
        list_pvec_include = []
        coef_p_t = []
        onlist = []
        onidx_list = []
        C_op_total = FermionOperator()
       #Kick out Pvec and pvec if |P_N| > T_N
        for iSD in range(len(list_Pvec)):
            Pvec = list_Pvec[iSD]
            pvec = list_pvec[iSD]
            l_include = True
            CoupCoef = 1.0
            for orb in range(N_open):
                if abs(Pvec[orb]) > Tvec[orb]:
                    l_include = False
                    break
                Tn_by2, Pn_by2, tn_by2, pn_by2 = Tvec[orb], Pvec[orb], tvec[orb], pvec[orb]
                CoupCoef *= SD_CSF_CoupCoef(Tn_by2, Pn_by2, tn_by2, pn_by2)
            if not l_include: continue
            list_Pvec_include.append(Pvec)
            list_pvec_include.append(pvec)
            coef_p_t.append(CoupCoef)
            onvec = np.zeros([2*N_open])
            for orb in range(N_open):
                if pvec[orb] == 1:
                    onvec[2*orb]   = 1.0
                    a_dagger = FermionOperator((2*orb,1),1.0)
                else:
                    onvec[2*orb+1] = 1.0
                    a_dagger = FermionOperator((2*orb+1,1),1.0)
                if orb == 0:
                    C_op = a_dagger
                else:
                    C_op = a_dagger*C_op
            onlist.append(onvec)
            onidx_list.append(get_on_idx(onvec))
            C_op *= CoupCoef
            C_op_total += C_op

        coef_p_t = np.array(coef_p_t)
        list_CSF.append([onlist,onidx_list,coef_p_t])
        list_C_op.append(C_op_total)

    if debug: check_spin_adapted_CSF(list_CSF,N_alpha-N_beta)

    if debug:
        print('\nlist_CSF in geneological_SD_CSF:')
        for iCSF, CSF in enumerate(list_CSF):
            print(f'\nCSF{iCSF}')
            print(CSF)
            print(f'\nTvec: {list_Tvec[iCSF]}')
            if l_get_op:
                print('\nCreation operator:')
                print(list_C_op[iCSF])


    if not l_get_op:
        return list_CSF
    else:
        return list_CSF, list_C_op


def check_spin_adapted_CSF(list_CSF,S_by2):
    """
    Check whether the list of CSFs generated by geneological coupling of SDs
    are eigenstates of S^2 and Sz with appropriate eigenvalue. Also check
    their orthonormality
    """

    n_spinorb = len(list_CSF[0][0][0])
   #n_open = N_alpha + N_beta
   #n_spinorb = 2*n_open

   #S_by2 = N_alpha - N_beta
    M_eigval = float(S_by2)*0.5
    S_sq_eigval = float(S_by2)*0.5*(float(S_by2)*0.5+1.0)

    Ssq_op = get_S_squared(n_spinorb)
    Sz_op  = get_S_z(n_spinorb)

    #Check orthonormality
    for iCSF, CSF_bra in enumerate(list_CSF):
        for jCSF in range(iCSF,len(list_CSF)):
            CSF_ket = list_CSF[jCSF]
            Selm = overlap_LCSD(CSF_bra[0],CSF_bra[1],CSF_bra[2],CSF_ket[0],CSF_ket[1],CSF_ket[2])
            if iCSF == jCSF and not np.isclose(Selm,1.0):
                print(f'CSF {iCSF} not normalized: {Selm}. Bombing out!')
                sys.exit()
            if iCSF != jCSF and not np.isclose(Selm,0.0):
                print(f'CSFs {iCSF, jCSF} not orthogonal: {Selm}. Bombing out!')
                sys.exit()

        lSsqeig, Ssqeig = judge_eigen_on_list(Ssq_op,CSF_bra[0],CSF_bra[1],CSF_bra[2],S_sq_eigval)
        lSzeig, Szeig = judge_eigen_on_list(Sz_op,CSF_bra[0],CSF_bra[1],CSF_bra[2],M_eigval)
        if not lSsqeig:
            print(f'CSF {iCSF} is not an eigenstate of S^2. Bombing out')
            for iSD in range(len(CSF_bra[0])):
                print(CSF_bra[2][iSD],CSF_bra[0][iSD])
            sys.exit()
        if not lSzeig:
            print(f'CSF {iCSF} is not an eigenstate of Sz. Bombing out')
            for iSD in range(len(CSF_bra[0])):
                print(CSF_bra[2][iSD],CSF_bra[0][iSD])
            sys.exit()

def create_missing_axial_sym_CSFs(list_CSF,list_SOMO_DMO,list_degmo):
    """
    Create the missing symmetry partners of CSFs
    """

    list_extra_CSF = []
    list_insert_index = []
    list_extra_somo_dmo = []
    nCSF = len(list_CSF)
    for iCSF in range(nCSF):
        tuple_dmo  = list_SOMO_DMO[iCSF][1]
        tuple_somo = list_SOMO_DMO[iCSF][0]
        list_dmo_ex = []
        for [degmo1, degmo2] in list_degmo:
            if degmo1 in tuple_somo or degmo2 in tuple_somo: continue
            if degmo1 in tuple_dmo and degmo2 not in tuple_dmo:
                list_dmo_ex.append([degmo1, degmo2])
            elif degmo1 not in tuple_dmo and degmo2 in tuple_dmo:
                list_dmo_ex.append([degmo2, degmo1])

        if len(list_dmo_ex) != 0:
            print(f'CSF Basis {iCSF}, SOMO: {list_SOMO_DMO[iCSF][0]}, DMO: {list_SOMO_DMO[iCSF][1]}')
            print(f'sym pair to be created, pair excitation: {list_dmo_ex}')
            list_insert_index.append(iCSF+1)
            CSF_sym_partner = copy.deepcopy(list_CSF[iCSF])
            [tuple_somo,tuple_dmo] = copy.deepcopy(list_SOMO_DMO[iCSF])
            list_dmo = list(tuple_dmo)
            for [dmo, vmo] in list_dmo_ex:
                list_dmo[list_dmo.index(dmo)] = vmo
            tuple_dmo = tuple(list_dmo)
            list_extra_somo_dmo.append([tuple_somo,tuple_dmo])
            print(f'CSF sym partner, SOMO: {tuple_somo}, DMO: {tuple_dmo}')
            for ion, onvec in enumerate(CSF_sym_partner[0]):
                for [dmo, vmo] in list_dmo_ex:
                    onvec[2*dmo:2*dmo+2] = 0.0
                    onvec[2*vmo:2*vmo+2] = 1.0
                CSF_sym_partner[1][ion] = get_on_idx(onvec)
            list_extra_CSF.append(CSF_sym_partner)

    for iCSF_extra in range(len(list_extra_CSF)-1,-1,-1):
        print(iCSF_extra)
        list_CSF.insert(list_insert_index[iCSF_extra],list_extra_CSF[iCSF_extra])
        list_SOMO_DMO.insert(list_insert_index[iCSF_extra],list_extra_somo_dmo[iCSF_extra])

   #print(f'List of SOMO and DMO after symmetry partner insertions.')
   #for ii, SOMO_DMO in enumerate(list_SOMO_DMO):
   #    print(f'CSF Basis {ii}, SOMO: {SOMO_DMO[0]}, DMO: {SOMO_DMO[1]}')

def reorder_list_CSF_for_sym(list_CSF,list_SOMO_DMO,Enuc,obt,tbt,nparal=1,debug=False):
    """
    Reorder list_CSF so that symmetry partners are adjacent
    """

    list_E_CSF = []
    list_doubly_degen_ind = []
    for CSF in list_CSF:
        E_CSF = Helm_between_CSFs(Enuc,obt,tbt,CSF,CSF)
        list_E_CSF.append(E_CSF)
    new_order = []
    degen_thrsh = 1e-8
    for iCSF in range(len(list_CSF)):
        if iCSF in new_order: continue
        new_order.append(iCSF)
        list_sym_partner = []
        for jCSF in range(iCSF+1,len(list_CSF)):
           #if np.isclose(list_E_CSF[iCSF],list_E_CSF[jCSF]): #np.isclose misjudge non-degen states as deg states
            if abs(list_E_CSF[iCSF]-list_E_CSF[jCSF]) < degen_thrsh:
                list_sym_partner.append(jCSF)

        if len(list_sym_partner) > 1:
            print(f'Strange. More than 1 partner is found for CSF{iCSF}: {list_sym_partner}')
            print('Bombing out!')
            sys.exit()
        elif len(list_sym_partner) == 1:
            new_order.append(list_sym_partner[0])

    print('new order of CSF with sym partners adjacent:')
    print(new_order)

    list_CSF = [list_CSF[ii] for ii in new_order]
    list_SOMO_DMO = [list_SOMO_DMO[ii] for ii in new_order]
    list_E_CSF = [list_E_CSF[ii] for ii in new_order]

    print(f'SOMOs and DMOs of the sym-reordered CSFs:')
    for ii, SOMO_DMO in enumerate(list_SOMO_DMO):
        E_CSF = Helm_between_CSFs(Enuc,obt,tbt,list_CSF[ii],list_CSF[ii])
        print(f'CSF Basis {ii}, SOMO: {SOMO_DMO[0]}, DMO: {SOMO_DMO[1]}, E: {E_CSF}')

    if nparal > 1:
        Hmat_CSF = construct_Hmat_CSFs_paral_triu(list_CSF,Enuc,obt,tbt,nparal)
    else:
        Hmat_CSF = construct_Hmat_CSFs(list_CSF,Enuc,obt,tbt)
    print('\nHamiltonian matrix of symmetry-ordered CSFs:')
    print_matrix(Hmat_CSF)
    Hmat_CSF_sparse = csr_matrix(Hmat_CSF)
    E_GS_CSF, psi_GS_CSF = get_ground_state(Hmat_CSF_sparse)
    print(f'\nE0 of symmetry-reordered CSFs: {E_GS_CSF}')
    print(f'psi_GS: {psi_GS_CSF}')

    nCSF = len(list_CSF)
    list_sym_sign = [0]*nCSF
    for iCSF in range(len(psi_GS_CSF)-1):
       #if np.isclose(list_E_CSF[iCSF],list_E_CSF[iCSF+1]):
        if abs(list_E_CSF[iCSF]-list_E_CSF[iCSF+1]) < degen_thrsh:
            list_sym_sign[iCSF] = round(np.sign(psi_GS_CSF[iCSF]) * np.sign(psi_GS_CSF[iCSF+1]))

    if debug:
        print(f'\nlist_sym_sign: {list_sym_sign}')

    list_sym_CSF = []
    for iCSF in range(len(list_sym_sign)):
        if list_sym_sign[iCSF] == 1 or list_sym_sign[iCSF] == -1:
            # Use the full GS projection directly (same principle as the mo_ml path in
            # select_ia_pairs_for_CSF_Eavg_with_sym_CSF): extract the two degenerate
            # components from psi_GS_CSF and normalise.  This gives the correct A1g
            # direction even when the fixed ±1/√2 heuristic would choose the wrong sign.
            _gs_raw = np.array([psi_GS_CSF[iCSF], psi_GS_CSF[iCSF + 1]])
            _nrm = np.linalg.norm(_gs_raw)
            if _nrm > 1e-12:
                lcvec = _gs_raw / _nrm
                print(f'  [sym_CSF] CSF{iCSF}+CSF{iCSF+1}: GS projection = {_gs_raw}, '
                      f'lcvec = {lcvec}')
            else:
                # Fallback (shouldn't happen for selected degenerate pairs)
                lcvec = np.array([np.sqrt(0.5), list_sym_sign[iCSF] * np.sqrt(0.5)])
                print(f'  [sym_CSF] CSF{iCSF}+CSF{iCSF+1}: GS norm~0, '
                      f'falling back to sign-based lcvec = {lcvec}')
            comb_CSF = LC_CSFs([list_CSF[iCSF],list_CSF[iCSF+1]],lcvec,True)
            list_sym_CSF.append(comb_CSF)
        else:
            if iCSF == 0:
                list_sym_CSF.append(list_CSF[iCSF])
            elif list_sym_sign[iCSF-1] == 1 or list_sym_sign[iCSF-1] == -1:
                continue
            else:
                list_sym_CSF.append(list_CSF[iCSF])

    n_symCSF = len(list_sym_CSF)
    Smat_CSF_symCSF = csr_matrix((nCSF,n_symCSF))
    for iCSF in range(nCSF):
        CSF = list_CSF[iCSF]
        for jsymCSF in range(n_symCSF):
            symCSF = list_sym_CSF[jsymCSF]
            Selm = overlap_CSFs(CSF,symCSF)
            if not np.isclose(Selm,0.0): Smat_CSF_symCSF[iCSF,jsymCSF] = Selm

    print('\nOverlap matrix between CSFs and sym-adapted CSFs:')
    print(Smat_CSF_symCSF)

    return list_CSF, list_SOMO_DMO, list_sym_sign, list_sym_CSF

#def opt_U_overlap(list_list_ia,list_list_theta,list_list_genmat,list_start_end,psi_GS_full,conv=1.0e-4,list_inivec=[],nparal=1,l_use_decomp_genmat=False,ldisp=False,debug=False):
    """
    Opt U for each CSF to maximize overlap with the ground state of the full pair ex space
    When l_use_decomp_genmat = True, list_list_genmat contains decomposed genmats, which give analytical U matrix
    """

    if debug: print('\nIn opt_U_overlap')

    n_CSF = len(list_list_ia)

    list_list_theta_opt = []
    list_Uvec_opt = []
    list_Umat_opt = []
    l_create_inivec = False
    if len(list_inivec) == 0: l_create_inivec = True
    for iCSF in range(n_CSF):
        print(list_start_end[iCSF][0],list_start_end[iCSF][1])
        target_vec = psi_GS_full[list_start_end[iCSF][0]:list_start_end[iCSF][1]]
        print(f'target vector of UCSF{iCSF}')
        print(target_vec)
        list_genmat = list_list_genmat[iCSF]
        list_theta = list_list_theta[iCSF]
        if l_create_inivec:
            inivec = np.zeros(len(target_vec))
            inivec[0] = 1.0
        else:
            inivec = list_inivec[iCSF]

        assert len(inivec) == len(target_vec)
        if len(list_genmat) != 0:
            list_theta_opt, opt_fun, Umat_opt, Uvec_opt = opt_U_overlap_one_UCSF(list_genmat,list_theta,target_vec,conv,inivec,nparal,l_use_decomp_genmat,ldisp,debug)
           #if 1.0+opt_fun > small:
           #    list_genmat_2nd = list_genmat + [list_genmat[0]]
           #    list_theta_2nd = copy.deepcopy(list_theta_opt)
           #    list_theta_2nd = np.append(list_theta_2nd,[0.0])
           #    print(f'Optimizing overlap using extra length of U train')
           #    print(list_theta_2nd)
           #    list_theta_opt, opt_fun, _, Uvec_opt = opt_U_overlap_one_UCSF(list_genmat_2nd,list_theta_2nd,target_vec,ldisp,debug)
           #    sys.exit()
        else:
            list_theta_opt = []
           #Uvec_opt = np.array([1.0])
           #Use normalized inivec as Uvec_opt for the case of no ia
            Uvec_opt = copy.deepcopy(inivec / np.linalg.norm(inivec))
            Umat_opt = np.eye(len(inivec))
        list_list_theta_opt.append(list_theta_opt)
        list_Uvec_opt.append(Uvec_opt)
        list_Umat_opt.append(Umat_opt)

    return list_list_theta_opt, list_Uvec_opt, list_Umat_opt

def opt_U_overlap(list_list_ia,list_list_theta,list_list_genmat,list_start_end,psi_GS_full,conv=1.0e-4,list_inivec=[],nparal=1,l_use_decomp_genmat=False,ldisp=False,debug=False):
    """
    Opt U for each CSF to maximize overlap with the ground state of the full pair ex space
    When l_use_decomp_genmat = True, list_list_genmat contains decomposed genmats, which give analytical U matrix
    """

    if debug: print('\nIn opt_U_overlap')

    n_CSF = len(list_list_ia)

    list_list_theta_opt = []
    list_Uvec_opt = []
    list_Umat_opt = []
    l_create_inivec = False
    if len(list_inivec) == 0: l_create_inivec = True
    for iCSF in range(n_CSF):
        print(list_start_end[iCSF][0],list_start_end[iCSF][1])
        target_vec = psi_GS_full[list_start_end[iCSF][0]:list_start_end[iCSF][1]]
        print(f'target vector of UCSF{iCSF}')
        print(target_vec)
        list_genmat = list_list_genmat[iCSF]
        list_theta = list_list_theta[iCSF]
        ndim_total = len(target_vec)
        padded_genmats = []
        for G in list_genmat:
            if G.shape[0] != ndim_total:
                # Expand the sparse matrix to the new total dimension
                # The extra rows/columns will be zeros
                from scipy import sparse
                G_padded = sparse.csr_matrix((ndim_total, ndim_total))
                # Insert the original matrix into the top-left corner
                G_padded[:G.shape[0], :G.shape[1]] = G
                padded_genmats.append(G_padded)
            else:
                padded_genmats.append(G)
        list_genmat = padded_genmats
        if l_create_inivec:
            inivec = np.zeros(len(target_vec))
            inivec[0] = 1.0
        else:
            inivec = list_inivec[iCSF]

        assert len(inivec) == len(target_vec)
        if len(list_genmat) != 0:
            list_theta_opt, opt_fun, Umat_opt, Uvec_opt = opt_U_overlap_one_UCSF(list_genmat,list_theta,target_vec,conv,inivec,nparal,l_use_decomp_genmat,ldisp,debug)
           #if 1.0+opt_fun > small:
           #    list_genmat_2nd = list_genmat + [list_genmat[0]]
           #    list_theta_2nd = copy.deepcopy(list_theta_opt)
           #    list_theta_2nd = np.append(list_theta_2nd,[0.0])
           #    print(f'Optimizing overlap using extra length of U train')
           #    print(list_theta_2nd)
           #    list_theta_opt, opt_fun, _, Uvec_opt = opt_U_overlap_one_UCSF(list_genmat_2nd,list_theta_2nd,target_vec,ldisp,debug)
           #    sys.exit()
        else:
            list_theta_opt = []
           #Uvec_opt = np.array([1.0])
           #Use normalized inivec as Uvec_opt for the case of no ia
            Uvec_opt = copy.deepcopy(inivec / np.linalg.norm(inivec))
            Umat_opt = np.eye(len(inivec))
        list_list_theta_opt.append(list_theta_opt)
        list_Uvec_opt.append(Uvec_opt)
        list_Umat_opt.append(Umat_opt)

    return list_list_theta_opt, list_Uvec_opt, list_Umat_opt

def opt_U_overlap_with_noEX_in_CAS(list_list_ia,list_list_theta,list_list_genmat,list_UCSF_subspace_start_end,psi_GS,Uopt_thrsh,list_list_pairex_CSF_vec,l_use_decomp_genmat,conv=1.0e-5,nparal=1,ldisp=False,debug=False):
    """
    Fit U theta parameters for U that contain only excitations out of CAS.
    """

    if debug: print('\nIn opt_U_overlap_with_noEX_in_CAS')

    n_refCSF = len(list_list_ia)
    ndim_full = list_UCSF_subspace_start_end[-1][-1]
    list_list_theta_opt = []
    list_list_Uvec_full = []
    list_list_vec_full = []
    n_Uvec_total = 0
    for iref in range(n_refCSF):
        [istart,iend] = list_UCSF_subspace_start_end[iref]
       #ref_CSF_vec = list_ref_CSF_vec[iref]
        ref_CSF_vec = list_list_pairex_CSF_vec[iref][0]
        assert len(ref_CSF_vec) == iend - istart
        list_theta = list_list_theta[iref]
        if len(list_theta) == 0:
            list_list_theta_opt.append([])
            list_Uvec_full = []
            for CSFvec in list_list_pairex_CSF_vec[iref]:
                Uvec_full = np.zeros(ndim_full)
                Uvec_full[istart:iend] = CSFvec
                n_Uvec_total += 1
                list_Uvec_full.append(Uvec_full)

            list_list_Uvec_full.append(list_Uvec_full)
            list_list_vec_full.append(list_Uvec_full)
           #print(f'\nUpdated list_list_theta_opt for iref {iref} without opt')
           #print(f'{len(list_list_theta_opt)}')
            continue

        list_vec_full = []
        for CSFvec in list_list_pairex_CSF_vec[iref]:
            vec_full = np.zeros(ndim_full)
            vec_full[istart:iend] = CSFvec
            list_vec_full.append(vec_full)
        list_list_vec_full.append(list_vec_full)

        list_genmat = list_list_genmat[iref]
        target_vec = psi_GS[istart:iend]
        target_norm = np.linalg.norm(target_vec)
        print(f'\nNorm of the target vector for iref {iref}: {target_norm}')
        norm_target_vec = copy.deepcopy(target_vec) / target_norm
       #list_inivec = [list_ref_CSF_vec[iref]] + list_list_pairex_CSF_vec[iref]
        list_inivec = list_list_pairex_CSF_vec[iref]
        x0 = np.array(copy.deepcopy(list_theta))

        def cost(x):
            return func_project_psi_into_U_space(x,list_genmat,list_inivec,norm_target_vec,l_use_decomp_genmat)

        options = {
        'maxiter' : 10000,
        'disp'    : ldisp,
        'gtol'    : 1.0e-4
        }

        tic = time.perf_counter()
        test_fun = cost(x0)
        toc = time.perf_counter()
        print(f'test_fun = {test_fun}, time for evaluation: {toc - tic}')

        tic = time.perf_counter()
       #sol = minimize(cost, x0, method='BFGS',options=options)
        arguments = (list_genmat,list_inivec,norm_target_vec,nparal,l_use_decomp_genmat)
        sol = minimize(grad_project_psi_into_U_space, x0, args=arguments, method='BFGS',options=options,jac=True)
        toc = time.perf_counter()
        print(f'Time for minimization: {toc - tic}')

        opt_fun1 = sol.fun
        theta_opt1 = sol.x

        opt_fun = opt_fun1
        theta_opt = theta_opt1
        nround = 1
        max_round = 1
        while (1.0 + opt_fun > conv) and nround < max_round:
            nround += 1
            print(f'\nRound {nround} optimization')
            list_genmat_xN = list_genmat*nround
            x0 = np.random.uniform(low=-0.5, high=-0.5, size = (nround*len(list_theta),))
            def cost_xn(x):
                return func_project_psi_into_U_space(x,list_genmat_xN,list_inivec,norm_target_vec,l_use_decomp_genmat)

            tic = time.perf_counter()
            test_fun = cost_xn(x0)
            toc = time.perf_counter()
            print(f'test_fun = {test_fun}, time for evaluation: {toc - tic}')

            tic = time.perf_counter()
           #sol_xn = minimize(cost_xn, x0, method='BFGS',options=options)
            arguments = (list_genmat_xN,list_inivec,norm_target_vec,nparal,l_use_decomp_genmat)
            sol_xn = minimize(grad_project_psi_into_U_space, x0, args=arguments, method='BFGS',options=options,jac=True)
            toc = time.perf_counter()
            print(f'Time for minimization sol_xn: {toc - tic}')
            opt_fun = sol_xn.fun
            theta_opt = sol_xn.x

        if 1.0 + opt_fun > conv:
            print(f'After {nround} of extension of theta list, convergence is not reached')
            print(f'Still moving on')
        else:
            print(f'Convergence is reached with {nround} train of U')

        inivec = list_inivec[0]
        if nround == 1:
            list_genmat_final = list_genmat
        else:
            list_genmat_final = list_genmat_xN
        if abs(opt_fun) < abs(opt_fun1):
            print(f'Theta list of the first round opt is taken')
            theta_opt = theta_opt1
            list_genmat_final = list_genmat

        list_list_theta_opt.append(theta_opt)
       #print(f'\nUpdated list_list_theta_opt for iref {iref} after opt')
       #print(f'{len(list_list_theta_opt)}')
        if l_use_decomp_genmat:
            Umat = make_Umat_decomp_genmat(list_genmat_final,theta_opt)
        else:
            Umat, _ = make_and_apply_U_matrix(list_genmat_final,theta_opt,True,inivec,False)

        list_Uvec = []
        for vec in list_inivec:
            Uvec = Umat@vec
            Uvec_full = np.zeros(ndim_full)
            Uvec_full[istart:iend] = Uvec
            n_Uvec_total += 1
            list_Uvec.append(Uvec_full)

        list_list_Uvec_full.append(list_Uvec)

    Umat_total = np.zeros([ndim_full,n_Uvec_total])
    Vmat_total = np.zeros([ndim_full,n_Uvec_total])
    icol = -1
    for list_Uvec_full in list_list_Uvec_full:
        for Uvec_full in list_Uvec_full:
            icol += 1
            Umat_total[:,icol] = Uvec_full

    icol = -1
    for list_vec_full in list_list_vec_full:
        for vec_full in list_vec_full:
            icol += 1
            Vmat_total[:,icol] = vec_full

    return list_list_theta_opt, list_list_Uvec_full, Umat_total, Vmat_total

def make_Uvec_full(list_list_ia,list_list_theta,l_use_decomp_genmat,list_list_genmat,list_UCSF_subspace_start_end,list_list_pairex_CSF_vec,list_list_pair_ex_space,debug=False):
    """
    Given the ia pairs, their theta, and the genmats in input, produce the Uvec for the reference CSF
    in the full space with all ex space
    """

    if debug: print('\nIn make_Uvec_full')
    n_ref = len(list_list_ia)
    ndim_full = list_UCSF_subspace_start_end[-1][-1]
    assert len(list_list_theta) == n_ref
    assert len(list_list_genmat) == n_ref
    assert len(list_UCSF_subspace_start_end) == n_ref
    assert len(list_list_pairex_CSF_vec) == n_ref

    list_list_Uvec_full = []
    list_list_vec_full = []
    list_list_refCSF = []
    list_list_Uext_mp2_CSF = []
    n_Uvec_total = 0
    for iref in range(n_ref):
        list_refCSF = []
        list_Uext_mp2_CSF = []
        [istart,iend] = list_UCSF_subspace_start_end[iref]
        ref_CSF_vec = list_list_pairex_CSF_vec[iref][0]
        assert len(ref_CSF_vec) == iend - istart
        list_theta = list_list_theta[iref]
        list_pairex_CSF_vec = list_list_pairex_CSF_vec[iref]
        list_Uvec_full = []
        if len(list_theta) == 0:
            for CSFvec in list_pairex_CSF_vec:
                refCSF = LC_CSFs(list_list_pair_ex_space[iref],CSFvec)
                list_refCSF.append(refCSF)
                list_Uext_mp2_CSF.append(refCSF)
                Uvec_full = np.zeros(ndim_full)
                Uvec_full[istart:iend] = CSFvec
                list_Uvec_full.append(Uvec_full)
                n_Uvec_total += 1

            list_list_Uvec_full.append(list_Uvec_full)
            list_list_vec_full.append(list_Uvec_full)
            list_list_refCSF.append(list_refCSF)
            list_list_Uext_mp2_CSF.append(list_Uext_mp2_CSF)
            continue

        list_genmat = list_list_genmat[iref]
        assert len(list_theta) == len(list_genmat)

        if l_use_decomp_genmat:
            Umat = make_Umat_decomp_genmat(list_genmat,list_theta)
        else:
            inivec = list_pairex_CSF_vec[0]
            Umat, _ = make_and_apply_U_matrix(list_genmat,list_theta,True,inivec,False)

        list_vec_full = []
        for vec in list_pairex_CSF_vec:
            ind_nonzero = np.where(vec != 0.0)[0]
            vec_short = vec[ind_nonzero]
            list_CSF_short = []
            for ind in ind_nonzero:
                list_CSF_short.append(list_list_pair_ex_space[iref][ind])
            refCSF = LC_CSFs(list_CSF_short,vec_short)
            list_refCSF.append(refCSF)
            vec_full = np.zeros(ndim_full)
            vec_full[istart:iend] = vec
            Uvec = Umat@vec
            Uext_CSF = LC_CSFs(list_list_pair_ex_space[iref],Uvec)
            list_Uext_mp2_CSF.append(Uext_CSF)
            Uvec_full = np.zeros(ndim_full)
            Uvec_full[istart:iend] = Uvec
            list_Uvec_full.append(Uvec_full)
            list_vec_full.append(vec_full)
            n_Uvec_total += 1

        list_list_Uvec_full.append(list_Uvec_full)
        list_list_vec_full.append(list_vec_full)
        list_list_refCSF.append(list_refCSF)
        list_list_Uext_mp2_CSF.append(list_Uext_mp2_CSF)

    Umat_total = np.zeros([ndim_full,n_Uvec_total])
    Vmat_total = np.zeros([ndim_full,n_Uvec_total])

    icol = -1
    for list_Uvec_full in list_list_Uvec_full:
        for Uvec_full in list_Uvec_full:
            icol += 1
            Umat_total[:,icol] = Uvec_full

    icol = -1
    for list_vec_full in list_list_vec_full:
        for vec_full in list_vec_full:
            icol += 1
            Vmat_total[:,icol] = vec_full

    return Umat_total, Vmat_total, list_list_refCSF, list_list_Uext_mp2_CSF

def opt_U_overlap_with_pair_ex_CSF(psi_GS,list_list_genmat,list_list_theta,list_list_CSF_ind_ex_space,list_UCSF_subspace_start_end,debug=False):
    """
    Fit U theta parameters for CSF that contain both CSFs from explicit pair excitations in active space and
    U pair excitation involving orbitals not in active space.
    """

    if debug: print('\nIn opt_U_overlap_with_pair_ex_CSF')

    nCSF_class = len(list_list_genmat)
    if len(list_list_theta) == 0:
        if debug: print('Initializing list_list_theta for fitting U')
        for iCSF_class in range(nCSF_class):
            list_list_theta.append([0.0]*len(list_list_genmat[iCSF_class]))
            if debug: print(list_list_theta[iCSF_class])

    list_opt_thetas = copy.deepcopy(list_list_theta)
    ndim_full = list_UCSF_subspace_start_end[-1][-1]
    list_list_Uvec = []
    zerovec_ndim_full = np.zeros(ndim_full)
    if debug: print(f'ndim_full: {ndim_full}')
    for iCSF_class in range(nCSF_class):
        [istart, iend] = list_UCSF_subspace_start_end[iCSF_class]
        list_Uvec = []
        if len(list_list_theta[iCSF_class]) == 0:
            for jj in range(istart,iend):
                uvec = copy.deepcopy(zerovec_ndim_full)
                uvec[jj] = 1.0
                list_Uvec.append(uvec)

            list_list_Uvec.append(list_Uvec)
            continue
        psi_target = psi_GS[istart:iend]
        psi_target = psi_target / np.linalg.norm(psi_target)
        print(istart,iend)
        list_genmat = list_list_genmat[iCSF_class]
       #print(len(list_genmat))
       #print(list_genmat)
        ndim = list_genmat[0].shape[0]
        assert iend - istart == ndim
        list_inivec = []
        for iCSF in range(len(list_list_CSF_ind_ex_space[iCSF_class])):
            print(iCSF,list_list_CSF_ind_ex_space[iCSF_class][iCSF])
            inivec = np.zeros(ndim)
            ind = list_list_CSF_ind_ex_space[iCSF_class][iCSF]
            inivec[ind] = 1.0
            list_inivec.append(inivec)
            for jCSF in range(iCSF):
                assert np.isclose(np.dot(list_inivec[jCSF],inivec),0.0)


        x0 = np.array(list_list_theta[iCSF_class])
        tic = time.perf_counter()
        test_fun = func_project_psi_into_U_space(x0,list_genmat,list_inivec,psi_target,l_use_decomp_genmat)
        toc = time.perf_counter()
        print(f'Time to evaluate function: {toc - tic}')
        print(f'test_fun = {test_fun}')
        def cost(x):
            return func_project_psi_into_U_space(x,list_genmat,list_inivec,psi_target,l_use_decomp_genmat)

        options = {
        'maxiter' : 10000,
        'disp'    : True,
        'gtol'    : 1.0e-4
        }

        sol = minimize(cost, x0, method='BFGS',options=options)

        Umat, Uvec = make_and_apply_U_matrix(list_genmat,sol.x)
        for inivec in list_inivec:
            Uvec = copy.deepcopy(zerovec_ndim_full)
            Uvec[istart:iend] = Umat@inivec
            list_Uvec.append(Uvec)


        list_opt_thetas[iCSF_class] = list(sol.x)
        list_list_Uvec.append(list_Uvec)

   #Check orthonormality between Uvec


    return list_opt_thetas, list_list_Uvec

def grad_project_psi_into_U_space(x,list_genmat,list_inivec,psi_target,nparal=1,l_use_decomp_genmat=False):

    l_paral = False
    if nparal > 1: l_paral = True

    list_grad_comp = []
    if l_paral:
       #Good Parallel. Commenting out the lines from Good Parallel to End of good parallel and commenting on
       #the subsequent list_grad_comp = Parallel(n_jobs=nparal) line returns to the code before good parallel.
        n_comp = len(x)+1
        chunk_size = n_comp // nparal
        list_all_idx = []
        for ii in range(n_comp): list_all_idx.append(ii)
        chunks = []
        for ii in range(0,nparal):
            chunks.append(list_all_idx[ii*chunk_size:(ii+1)*chunk_size])
        chunks[-1] += list_all_idx[nparal*chunk_size:]
        def grad_chunk_comp_prj_psi_to_U_space(list_idx,x,list_genmat,list_inivec,psi_target,l_use_decomp_genmat):
            list_grad_comp_1chunk = []
            for icomp in list_idx:
                grad_comp = grad_comp_prj_psi_to_U_space(icomp,x,list_genmat,list_inivec,psi_target,l_use_decomp_genmat)
                list_grad_comp_1chunk.append(grad_comp)

            return list_grad_comp_1chunk

        list_list_grad_comp = Parallel(n_jobs=nparal)(delayed(grad_chunk_comp_prj_psi_to_U_space)(chunk,x,list_genmat,list_inivec,psi_target,l_use_decomp_genmat) for chunk in chunks)
        list_grad_comp = []
        for item in list_list_grad_comp: list_grad_comp += item
       #End of good parallel
       #list_grad_comp = Parallel(n_jobs=nparal)(delayed(grad_comp_prj_psi_to_U_space)(ii,x,list_genmat,list_inivec,psi_target,l_use_decomp_genmat) for ii in range(len(x)+1))
    else:
        for icomp in range(len(x)+1):
            grad_comp = grad_comp_prj_psi_to_U_space(icomp,x,list_genmat,list_inivec,psi_target,l_use_decomp_genmat)
            list_grad_comp.append(grad_comp)

    funval = list_grad_comp[0]
    grad = np.array(list_grad_comp[1:])

    return funval,grad

def grad_comp_prj_psi_to_U_space(icomp,x,list_genmat,list_inivec,psi_target,l_use_decomp_genmat=False):

    eps = 1.0e-4
    if icomp == 0:
        prjtion = func_project_psi_into_U_space(x,list_genmat,list_inivec,psi_target,l_use_decomp_genmat)
        return prjtion
    else:
        x_disp = x.copy()
        x_disp[icomp-1] = x[icomp-1] + eps
        prj_plus  = func_project_psi_into_U_space(x_disp,list_genmat,list_inivec,psi_target,l_use_decomp_genmat)
        x_disp[icomp-1] = x[icomp-1] - eps
        prj_minus = func_project_psi_into_U_space(x_disp,list_genmat,list_inivec,psi_target,l_use_decomp_genmat)
        grad_comp = (prj_plus - prj_minus) / (2.0*eps)
        return grad_comp


def func_project_psi_into_U_space(x,list_genmat,list_inivec,psi_target,l_use_decomp_genmat=False):
    """
    Projection of psi_target onto a set of {U|vec>} states as function of the theta angles
    in U
    """

    inivec = list_inivec[0]
    if l_use_decomp_genmat:
         Umat = make_Umat_decomp_genmat(list_genmat,x)
         Uvec = Umat@inivec
    else:
        Umat, Uvec = make_and_apply_U_matrix(list_genmat,x,True,inivec,False)
   #for inivec in list_inivec:
    norm_val = np.linalg.norm(Uvec)
    if not np.isclose(norm_val, 1.0):
        print(f"DEBUG: Norm is {norm_val} for group with dim {len(Uvec)}")
        Uvec = Uvec / norm_val

    assert np.isclose(np.linalg.norm(Uvec),1.0)
    proj_val = (np.dot(Uvec,psi_target))**2.0
    for inivec in list_inivec[1:]:
        Uvec = Umat@inivec
        proj_val += (np.dot(Uvec,psi_target))**2.0

    proj_val *= -1.0 #To convert maximization to minimization
    if abs(proj_val) > 1.0 + 1.0e-8: #Give it some small room
        print(f'proj_val {proj_val} has magnitude > 1. Bombing out!')
        sys.exit()

    return proj_val


def opt_U_overlap_one_UCSF(list_genmat,list_theta,target_vec,conv,inivec,nparal=1,l_use_decomp_genmat=False,ldisp=False,debug=False):
    """
    Opt U for one CSF to maximize overlap with a target vector
    When l_use_decomp_genmat = True, list_genmat contains decomposed genmat that give analytical Umat
    """

   #ndim = list_genmat[0].shape[0]
   #assert ndim == len(target_vec)
    ndim = len(target_vec)
   #print(f'length of list_genmat: {len(list_genmat)}')
   #print(f'length of list_theta: {len(list_theta)}')
   #print(list_theta)
    assert len(list_genmat) == len(list_theta)

   #normalize the target_vec
    target_norm = np.linalg.norm(target_vec)
    norm_target = target_vec / target_norm
    print(f'target norm: {target_norm}')
    print(f'norm_target: {norm_target}')
   #inivec provded in input
   #inivec = np.zeros(ndim)
   #inivec[0] = 1.0

    def cost(x):
        return eval_ref_UCSF_ovlp(x, list_genmat,norm_target,inivec,l_use_decomp_genmat)

    options = {
        'maxiter' : 10000,
        'disp'    : ldisp,
        'gtol'    : 1.0e-4
    }

    x0 = np.array(copy.deepcopy(list_theta))
    if np.isclose(np.sum(x0*x0),0.0):
        x0 =  np.random.uniform(low=-0.5, high=-0.5, size = (len(list_theta),))
    tic = time.perf_counter()
    ovlp_initial = cost(x0)
    toc = time.perf_counter()
    if debug:
        print(f'Time to evaluate function: {toc-tic}')
        print(f'\ncost(x0) = {ovlp_initial}')


   #sol = minimize(cost, x0, method='BFGS',options=options)
    arguments = (list_genmat,norm_target,inivec,nparal,l_use_decomp_genmat)
   #sol = minimize(grad_ref_UCSF_ovlp, x0, args=arguments, method='BFGS',options=options,jac=True)
    sol = minimize(grad_ref_UCSF_ovlp_no_data_share, x0, args=arguments, method='BFGS',options=options,jac=True)
   #arguments_paral = (list_genmat,norm_target,inivec,l_use_decomp_genmat)
   #print(f'minimize_parallel with max_workers: {nparal}')
   #sol_paral = minimize_parallel(eval_ref_UCSF_ovlp,x0,args=arguments_paral,parallel={'loginfo': True, 'max_workers': nparal})

    if l_use_decomp_genmat:
        Umat_opt = make_Umat_decomp_genmat(list_genmat,sol.x)
        Uvec_opt = Umat_opt@inivec
    else:
        Umat_opt, Uvec_opt = make_and_apply_U_matrix(list_genmat,sol.x,True,inivec)
    theta_opt = sol.x
    fun_opt = sol.fun
    fun_opt_1st = fun_opt
    theta_opt_1st = theta_opt
    Umat_opt_1st = Umat_opt
    Uvec_opt_1st = Uvec_opt
    print(f'Uvec_opt_1st: {Uvec_opt_1st}')
    print(f'theta_opt after 1st round fitting: {theta_opt}')
    #print(f'\nOverlap between opt Uvec and normalized target vec: {np.dot(Uvec_opt,norm_target)}')

    small = conv

    nround = 1
    max_nround = 10
    while target_norm*(1.0+fun_opt) > small and nround <= max_nround:
        nround += 1
       #if nround > max_nround:
       #    print(f'No convergence up to {max_nround} trains. Bombing out!')
       #    sys.exit()
        print(f'\nDo another round of U opt with simultaneous {nround} trains of U')
        x0 = np.random.uniform(low=-0.5, high=-0.5, size = (nround*len(list_theta),))
       #Randomization of the initial parameters is the key for the success of using two trains.
       #The following all zero initialization for two trains won't work. It gives the same result as one train.
       #This is because the two initial trains with all zero amplitudes commute. The gradients of the two sets
       #of parameters are identical.
       #x0 = np.array([0.0] * 2*len(list_theta)) #This won't work. Randomization of the initial parameters is the key.
        list_genmat_xn = list_genmat*nround
        def cost_Uxn(x):
            return eval_ref_UCSF_ovlp(x, list_genmat_xn,norm_target,inivec,l_use_decomp_genmat)
        tic = time.perf_counter()
        print(f'test_fun {nround} round: {cost_Uxn(x0)}')
        toc = time.perf_counter()
        print(f'Time to evaluate function: {toc - tic}')
       #sol_Uxn = minimize(cost_Uxn,x0,method='BFGS',options=options)
        arguments = (list_genmat_xn,norm_target,inivec,nparal,l_use_decomp_genmat)
       #sol_Uxn = minimize(grad_ref_UCSF_ovlp,x0,args=arguments, method='BFGS',options=options,jac=True)
        sol_Uxn = minimize(grad_ref_UCSF_ovlp_no_data_share,x0,args=arguments, method='BFGS',options=options,jac=True)
        print(f'Improvement with {nround} trains: {sol_Uxn.fun - fun_opt}')
        if l_use_decomp_genmat:
            Umat_opt = make_Umat_decomp_genmat(list_genmat_xn,sol_Uxn.x)
            Uvec_opt = Umat_opt@inivec
        else:
            Umat_opt, Uvec_opt = make_and_apply_U_matrix(list_genmat_xn,sol_Uxn.x,True,inivec)
        theta_opt = sol_Uxn.x
        print(f'theta_opt: {theta_opt}')
        Uvec_opt_print = Uvec_opt
        if np.dot(Uvec_opt,norm_target) < 0.0: Uvec_opt_print = - Uvec_opt
        print(f'Uvec_opt_print: {Uvec_opt_print}')
        print(f'Uvec_opt dot norm_target: {np.dot(Uvec_opt_print,norm_target)}')
        res_vec = norm_target - Uvec_opt_print
        print(f'residual vector: {res_vec}')
        print(f'Normalized res. vec: {res_vec / np.linalg.norm(res_vec)}')
        fun_opt = sol_Uxn.fun

    if nround > max_nround:
        print(f'Moving on although the U fitting does not converge.')
    else:
        print(f'Iterations on adding U trains converged: with {1.0+fun_opt} vs. threshold {small}')

    if abs(fun_opt - fun_opt_1st) < small:
        theta_opt = theta_opt_1st
        fun_opt = fun_opt_1st
        Umat_opt = Umat_opt_1st
        Uvec_opt = Uvec_opt_1st

    return theta_opt, fun_opt, Umat_opt, Uvec_opt

def _solver_options(method, maxiter=5000, tol=1.0e-4):
    """Per-method scipy.minimize options for a uniform convergence tolerance.
    Gradient-based methods take gtol; Powell (gradient-free) takes xtol/ftol."""
    opts = {'maxiter': maxiter}
    if method in ('BFGS', 'CG', 'L-BFGS-B', 'TNC', 'trust-constr', 'Newton-CG'):
        opts['gtol'] = tol
    elif method == 'Powell':
        opts['xtol'] = tol
        opts['ftol'] = tol
    elif method == 'Nelder-Mead':
        opts['xatol'] = tol
        opts['fatol'] = tol
    return opts

#def opt_theta_for_energy_UCSF_1(list_ex_states, initial_theta, list_genmat, l_use_decomp_genmat, Enuc, obt, tbt, nparal, method, no_evals=2, weight=[0.5, 0.5], conv_threshold=1e-3, debug=False):
    """
    Generate list_UCSF from list_CSF_ref, initial_theta, list_genmat. Then, build the UCSF Hamiltonian
    and use scipy.minimize() in order to variationally optimize initial_theta to
    minimize the cost function within conv_threshold: weight[0]E_{0}(initial_theta) + ... + weight[len(weight)-1]E_{len(weight)}(initial_theta).
    """
    if debug: print('\nIn opt_theta_for_energy_UCSF')

    sizes = [len(th) for th in initial_theta]
    offsets = np.cumsum([0] + sizes)
    x0 = np.concatenate([np.asarray(th, dtype=float) for th in initial_theta]) if sum(sizes) else np.array([])

    weight = np.asarray(weight, dtype=float)

    def unpack(xflat):
        return [xflat[offsets[g]:offsets[g + 1]] for g in range(len(sizes))]

    def f(xflat):
        theta = unpack(xflat)
        UCSF = [None] * len(list_ex_states)
        for iCSF_group in range(len(list_ex_states)):
            ndim = len(list_ex_states[iCSF_group])
            inivec = np.zeros(ndim)
            inivec[0] = 1.0

            print(f"group={iCSF_group}, ndim={ndim}, len(genmat)={len(list_genmat[iCSF_group])}")

            if len(list_genmat[iCSF_group]) == 0:
                Uvec = inivec.copy()
                Umat = np.eye(ndim)
            elif l_use_decomp_genmat:
                Umat = make_Umat_decomp_genmat(
                    list_genmat[iCSF_group],
                    theta[iCSF_group])
                Uvec = Umat @ inivec
                Uvec = Uvec / np.linalg.norm(Uvec)
            else:
                Umat, Uvec = make_and_apply_U_matrix(
                    list_genmat[iCSF_group],
                    theta[iCSF_group])

            UCSF[iCSF_group] = make_UCSF_state(list_ex_states[iCSF_group], Uvec)

        Hmat_UCSF = construct_Hmat_CSFs_paral_triu(UCSF, Enuc, obt, tbt, nparal)
        evals, _ = scipy.sparse.linalg.eigsh(csr_matrix(Hmat_UCSF), k=no_evals, which='SA')
        evals = np.sort(evals)
        return float(np.dot(weight, evals))

    if method is None:
        x_opt = x0
    else:
        res = minimize(f, x0, method=method, tol=conv_threshold,
                       options=_solver_options(method))
        x_opt = res.x

    theta_opt = unpack(x_opt)

    Umat_opt = []
    Uvec_opt = []
    UCSF_opt = []
    for iCSF_group in range(len(list_ex_states)):
        ndim = len(list_ex_states[iCSF_group])
        inivec = np.zeros(ndim)
        inivec[0] = 1.0

        if len(list_genmat[iCSF_group]) == 0:
            Uvec = inivec.copy()
            Umat = np.eye(ndim)
        elif l_use_decomp_genmat:
            Umat = make_Umat_decomp_genmat(
                list_genmat[iCSF_group],
                theta_opt[iCSF_group])
            Uvec = Umat @ inivec
            Uvec = Uvec / np.linalg.norm(Uvec)
        else:
            Umat, Uvec = make_and_apply_U_matrix(
                list_genmat[iCSF_group],
                theta_opt[iCSF_group])

        UCSF = make_UCSF_state(list_ex_states[iCSF_group], Uvec)
        UCSF_opt.append(UCSF)
        Umat_opt.append(Umat)
        Uvec_opt.append(Uvec)

    return theta_opt, Uvec_opt, Umat_opt, UCSF_opt

#def opt_theta_for_energy_UCSF_1(list_ex_states, initial_theta, list_genmat, l_use_decomp_genmat,Enuc, obt, tbt, nparal, method, list_theta_prev=None,no_evals=2, weight=[0.5, 0.5], conv_threshold=1e-3, debug=False):

    if list_theta_prev is None:
        list_theta_prev = []

    sizes = [len(th) for th in initial_theta]
    offsets = np.cumsum([0] + sizes)
    x0 = np.concatenate([np.asarray(th, dtype=float) for th in initial_theta]) if sum(sizes) else np.array([])
    weight = np.asarray(weight, dtype=float)

    def unpack(xflat):
        return [xflat[offsets[g]:offsets[g + 1]] for g in range(len(sizes))]

    def apply_genmat(list_genmat_group, theta_group, Uvec_in):
        """Apply one train's rotation to Uvec_in."""
        if len(list_genmat_group) == 0:
            return Uvec_in.copy()
        if l_use_decomp_genmat:
            Umat = make_Umat_decomp_genmat(list_genmat_group, theta_group)
            Uvec = Umat @ Uvec_in
        else:
            _, Uvec = make_and_apply_U_matrix(list_genmat_group, theta_group)
        return Uvec / np.linalg.norm(Uvec)

    def f(xflat):
        theta = unpack(xflat)
        UCSF = [None] * len(list_ex_states)
        for iCSF_group in range(len(list_ex_states)):
            ndim = len(list_ex_states[iCSF_group])
            inivec = np.zeros(ndim)
            inivec[0] = 1.0

            # Apply all previous fixed trains first
            Uvec = inivec.copy()
            for theta_prev in list_theta_prev:
                Uvec = apply_genmat(list_genmat[iCSF_group], theta_prev[iCSF_group], Uvec)

            # Apply current (new) train on top
            Uvec = apply_genmat(list_genmat[iCSF_group], theta[iCSF_group], Uvec)

            UCSF[iCSF_group] = make_UCSF_state(list_ex_states[iCSF_group], Uvec)

        Hmat_UCSF = construct_Hmat_CSFs_paral_triu(UCSF, Enuc, obt, tbt, nparal)
        evals, _ = scipy.sparse.linalg.eigsh(csr_matrix(Hmat_UCSF), k=no_evals, which='SA')
        evals = np.sort(evals)
        return float(np.dot(weight, evals))

    if method is None:
        x_opt = x0
    else:
        res = minimize(f, x0, method=method, tol=conv_threshold,
                       options=_solver_options(method))
        x_opt = res.x

    theta_opt = unpack(x_opt)

    # Build final Uvec, Umat, UCSF using all trains including current
    Umat_opt = []
    Uvec_opt = []
    UCSF_opt = []
    for iCSF_group in range(len(list_ex_states)):
        ndim = len(list_ex_states[iCSF_group])
        inivec = np.zeros(ndim)
        inivec[0] = 1.0

        Uvec = inivec.copy()
        for theta_prev in list_theta_prev:
            Uvec = apply_genmat(list_genmat[iCSF_group], theta_prev[iCSF_group], Uvec)
        Uvec = apply_genmat(list_genmat[iCSF_group], theta_opt[iCSF_group], Uvec)

        if len(list_genmat[iCSF_group]) == 0:
            Umat = np.eye(ndim)
        elif l_use_decomp_genmat:
            Umat = make_Umat_decomp_genmat(list_genmat[iCSF_group], theta_opt[iCSF_group])
        else:
            Umat, _ = make_and_apply_U_matrix(list_genmat[iCSF_group], theta_opt[iCSF_group])

        UCSF = make_UCSF_state(list_ex_states[iCSF_group], Uvec)
        UCSF_opt.append(UCSF)
        Umat_opt.append(Umat)
        Uvec_opt.append(Uvec)

    return theta_opt, Uvec_opt, Umat_opt, UCSF_opt

def opt_theta_for_energy_UCSF_non_group(list_ex_states, initial_theta, list_genmat, l_use_decomp_genmat,Enuc, obt, tbt, nparal, method,no_evals, weight, list_theta_prev=None, conv_threshold=1e-3, debug=False):

    weight = np.array([1 / no_evals] * no_evals)

    if list_theta_prev is None:
        list_theta_prev = []

    if debug:
        print("\n[opt_theta_for_energy_UCSF_1 with projected Hamiltonian + multi-train]")

    weight = np.asarray(weight, dtype=float)
    n_groups = len(list_ex_states)

    sizes = [len(th) for th in initial_theta]
    offsets = np.cumsum([0] + sizes)
    x0 = np.concatenate([np.asarray(th, dtype=float) for th in initial_theta]) if sum(sizes) else np.array([])

    def unpack(xflat):
        return [xflat[offsets[g]:offsets[g + 1]] for g in range(len(sizes))]

    def apply_genmat(list_genmat_group, theta_group, Uvec_in):
        """Apply one train's rotation to Uvec_in."""
        if len(list_genmat_group) == 0:
            return Uvec_in.copy()
        if l_use_decomp_genmat:
            Umat = make_Umat_decomp_genmat(list_genmat_group, theta_group)
            Uvec = Umat @ Uvec_in
        else:
            Umat, _ = make_and_apply_U_matrix(list_genmat_group, theta_group)
            Uvec = Umat @ Uvec_in
        return Uvec / np.linalg.norm(Uvec)

    # Build full CSF basis and start/end indices
    full_basis = []
    list_start_end = []
    start = 0
    for g in range(n_groups):
        states = list_ex_states[g]
        full_basis.extend(states)
        end = start + len(states)
        list_start_end.append((start, end))
        start = end

    if debug:
        print("Building full Hamiltonian once...")

    # Build full Hamiltonian once
    H_full = np.array(construct_Hmat_CSFs_paral_triu(full_basis, Enuc, obt, tbt, nparal))

    def build_Uvecs(theta):
        """Build Uvec for each group by composing all previous trains then current theta."""
        list_Uvec = []
        list_Umat = []
        for g in range(n_groups):
            ndim = len(list_ex_states[g])
            inivec = np.zeros(ndim)
            inivec[0] = 1.0

            # Apply all previous fixed trains first
            Uvec = inivec.copy()
            for theta_prev in list_theta_prev:
                Uvec = apply_genmat(list_genmat[g], theta_prev[g], Uvec)

            # Apply current train on top
            Uvec = apply_genmat(list_genmat[g], theta[g], Uvec)

            # Umat is only for current train's theta (not composed)
            if len(list_genmat[g]) == 0:
                Umat = np.eye(ndim)
            elif l_use_decomp_genmat:
                Umat = make_Umat_decomp_genmat(list_genmat[g], theta[g])
            else:
                Umat, _ = make_and_apply_U_matrix(list_genmat[g], theta[g])

            list_Uvec.append(Uvec)
            list_Umat.append(Umat)

        return list_Uvec, list_Umat

    def build_projected_matrices(theta):
        list_Uvec, _ = build_Uvecs(theta)

        H_eff = np.zeros((n_groups, n_groups))
        S_eff = np.zeros((n_groups, n_groups))

        for i in range(n_groups):
            ui = list_Uvec[i]
            si, ei = list_start_end[i]
            for j in range(i, n_groups):
                uj = list_Uvec[j]
                sj, ej = list_start_end[j]

                H_block = H_full[si:ei, sj:ej]
                Hij = ui @ (H_block @ uj)
                Sij = ui @ uj if i == j else 0.0

                H_eff[i, j] = Hij
                H_eff[j, i] = Hij
                S_eff[i, j] = Sij
                S_eff[j, i] = Sij

        return H_eff, S_eff, list_Uvec

    def f(xflat):
        theta = unpack(xflat)
        H_eff, S_eff, _ = build_projected_matrices(theta)
        evals = scipy.linalg.eigh(H_eff, S_eff, eigvals_only=True)
        evals = np.sort(evals)[:no_evals]
        return float(np.dot(weight, evals))

    if method is None or len(x0) == 0:
        x_opt = x0
    else:
        res = minimize(f, x0, method=method, tol=conv_threshold,
                       options=_solver_options(method))
        x_opt = res.x

    theta_opt = unpack(x_opt)
    list_Uvec_opt, list_Umat_opt = build_Uvecs(theta_opt)

    UCSF_opt = []
    for g in range(n_groups):
        UCSF = make_UCSF_state(list_ex_states[g], list_Uvec_opt[g])
        UCSF_opt.append(UCSF)

    return theta_opt, list_Uvec_opt, list_Umat_opt, UCSF_opt

def opt_multi_train_UCSF(list_list_pair_ex_space, initial_theta, list_genmat, l_use_decomp_genmat,
                          Enuc, obt, tbt, nparal, list_sym_CSF_vec, shared_unitary_groups, no_evals, list_list_extra_vecs=None, conv_threshold=1e-4, energy_threshold=1e-4, max_trains=1,
                          method="L-BFGS-B", debug=False, frozen_theta_map=None):

    if list_list_extra_vecs is None:
        list_list_extra_vecs = [[] for _ in range(len(list_list_pair_ex_space))]

    weight = [1 / no_evals] * no_evals

    list_list_list_theta = []
    list_Uvec_opt = []
    list_Umat_opt = []
    list_UCSF_opt = []
    prev_energy = None

    for k in range(max_trains):

        if debug:
            print(f"\n=== Train {k+1} ===")

        # In opt_multi_train_UCSF:
        initial_theta_k = initial_theta

        print(initial_theta_k)

        # Optimize kth train, passing all previous thetas as fixed
        theta_k, Uvec_opt, Umat_opt, UCSF_opt_k = opt_theta_for_energy_UCSF_group(
            list_list_pair_ex_space,
            initial_theta_k,
            list_genmat,
            l_use_decomp_genmat,
            Enuc, obt, tbt,
            nparal, method, list_sym_CSF_vec,
            no_evals,
            weight,
            shared_unitary_groups=shared_unitary_groups,
            list_theta_prev=None, list_list_extra_vecs=list_list_extra_vecs, conv_threshold=1e-3, debug=False,
            frozen_theta_map=frozen_theta_map)

        print(theta_k)

        list_list_list_theta.append(theta_k)

        # Compute energy from fully composed state
        Hmat = construct_Hmat_CSFs_paral_triu(UCSF_opt_k, Enuc, obt, tbt, nparal)
        evals, _ = scipy.sparse.linalg.eigsh(csr_matrix(Hmat), k=no_evals, which='SA')
        curr_energy = float(np.dot(weight, np.sort(evals)))

        list_Uvec_opt.append(Uvec_opt)
        list_Umat_opt.append(Umat_opt)
        list_UCSF_opt.append(UCSF_opt_k)

        if prev_energy is None:
            print(f"  Train {k+1}: E = {curr_energy:.10f}")
        else:
            dE = abs(curr_energy - prev_energy)
            print(f"  Train {k+1}: E = {curr_energy:.10f}  |ΔE| = {dE:.2e}")

        if prev_energy is not None and abs(curr_energy - prev_energy) < energy_threshold:
            print(f" Converged after {k+1} trains")
            break

        prev_energy = curr_energy

    final_list_list_theta_opt = list_list_list_theta[-1]  # last train's thetas, per group
    final_list_Uvec_opt = list_Uvec_opt[-1]  # last train's Uvecs, per group
    final_list_Umat_opt = list_Umat_opt[-1]  # last train's Umats, per group
    final_list_UCSF_opt = list_UCSF_opt[-1]  # last train's UCSFs, per group

    return final_list_list_theta_opt, final_list_Uvec_opt, final_list_Umat_opt, final_list_UCSF_opt

def opt_theta_for_energy_UCSF(list_list_ia,list_list_theta,list_list_genmat,shared_unitary_groups,psi_GS_full,Uopt_thrsh,
                              list_sym_CSF_vec,nparal,l_use_decomp_genmat,Hmat_full_space,no_evals,weight,conv_threshold=1e-4,method="BFGS",debug=False):
    """
    Generate theta-dependent UCSF vectors from list_list_theta and list_list_genmat.
    Project the prebuilt full-space Hamiltonian into the theta-dependent UCSF subspace:

        H_UCSF(theta) = V(theta)^T H_full V(theta)

    where each column of V(theta) is one UCSF vector embedded into the full CSF space.
    Then use scipy.minimize() to variationally optimize theta to minimize

        weight[0] E_0(theta) + ... + weight[len(weight)-1] E_{len(weight)-1}(theta).
    """

    weight = np.array([1 / no_evals] * no_evals)

    if debug:
        print('\nIn opt_theta_for_energy_UCSF')

    # Use thetas passed into the function as the initial guess
    sizes = [len(th) for th in list_list_theta]
    offsets = np.cumsum([0] + sizes)
    x0 = np.concatenate([np.asarray(th, dtype=float) for th in list_list_theta]) if sum(sizes) else np.array([])

    n_total = sum(sizes)
    weight = np.asarray(weight, dtype=float)

    if len(weight) != no_evals:
        raise ValueError(f'len(weight)={len(weight)} must equal no_evals={no_evals}')

    n_groups = len(list_list_ia)
    n_full = Hmat_full_space.shape[0]

    # Precompute initial basis vectors for each group
    list_inivec = []
    for iCSF_group in range(n_groups):
        ndim = len(list_list_ia[iCSF_group])
        inivec = np.zeros(ndim)
        inivec[0] = 1.0
        list_inivec.append(inivec)

    def unpack(xflat):
        return [xflat[offsets[g]:offsets[g + 1]] for g in range(len(sizes))]

    def build_projected_objects(theta):
        n_cols = sum([len(np.where(vec != 0.0)[0]) for vec in list_sym_CSF_vec])
        Vmat = np.zeros((n_full, n_cols))

        col_idx = 0
        for i_group in range(n_groups):
            Umat = make_Umat_decomp_genmat(list_list_genmat[i_group], theta[i_group])

            # Find all indices in this group that we marked as 'independent'
            # This is where the 'mask' from the previous step is used
            active_indices = np.where(list_sym_CSF_vec[i_group] != 0.0)[0]

            for idx in active_indices:
                # Create a unit vector for this specific configuration within the group
                local_vec = np.zeros(len(list_sym_CSF_vec[i_group]))
                local_vec[idx] = 1.0

                # Rotate it independently
                rotated_vec = Umat @ local_vec

                # Place it in its own unique column
                istart, iend = list_UCSF_subspace_start_end[i_group]
                Vmat[istart:iend, col_idx] = rotated_vec
                col_idx += 1

        return Vmat

    def f(xflat):
        theta = unpack(xflat)

        Vmat = build_projected_objects(theta)

        Hmat_UCSF = Vmat.transpose() @ Hmat_full_space @ Vmat

        evals = np.linalg.eigvalsh(Hmat_UCSF)
        evals = evals[:no_evals]

        return float(np.dot(weight, evals))

    if n_total == 0:
        theta_opt = unpack(x0)
    else:
        res = minimize(f, x0, method=method, tol=conv_threshold,
                       options=_solver_options(method))
        theta_opt = unpack(res.x)

    Umat_opt = []
    Uvec_opt = []
    UCSF_opt = []

    print('iCSF_group:', n_groups)
    print(theta_opt)

    for iCSF_group in range(n_groups):
        Umat = make_Umat_decomp_genmat(list_list_genmat[iCSF_group], theta_opt[iCSF_group])

        active_indices = np.where(list_sym_CSF_vec[iCSF_group] != 0.0)[0]

        for idx in active_indices:
            local_vec = np.zeros(len(list_sym_CSF_vec[iCSF_group]))
            local_vec[idx] = 1.0

            Uvec = Umat @ local_vec

            UCSF = make_UCSF_state(list_list_ia[iCSF_group], Uvec)

            UCSF_opt.append(UCSF)
            Uvec_opt.append(Uvec)

        Umat_opt.append(Umat)

    return theta_opt, Uvec_opt, Umat_opt, UCSF_opt

# def opt_theta_for_energy_UCSF_group(list_ex_states, initial_theta, list_genmat, l_use_decomp_genmat,
#                                     Enuc, obt, tbt, nparal, method, list_sym_CSF_vec,no_evals, weight, list_UCSF_subspace_start_end, list_theta_prev=None, list_list_extra_vecs=None,conv_threshold=1e-3, debug=False):
#
#     if list_list_extra_vecs is None:
#         list_list_extra_vecs = [[] for _ in range(n_groups)]
#
#     if list_theta_prev is None:
#         list_theta_prev = []
#
#     n_groups = len(list_ex_states)
#     n_cols = sum([len(np.where(mask != 0.0)[0]) for mask in list_sym_CSF_vec]) \
#              + sum([len(vecs) for vecs in list_list_extra_vecs])
#
#     sizes = [len(th) for th in initial_theta]
#     n_total = sum(sizes)
#     offsets = list(accumulate([0] + sizes))
#
#     print('offsets:', offsets)
#
#     x0 = (np.concatenate([np.atleast_1d(np.asarray(th, dtype=float)) for th in initial_theta]) if n_total > 0 else np.array([]))
#
#     print(x0)
#
#     def genmats_equal(gm1, gm2):
#         if len(gm1) != len(gm2):
#             return False
#         for g1, g2 in zip(gm1, gm2):
#             if not np.allclose(g1, g2):
#                 return False
#         return True
#
#     genmat_group_id = []
#     unique_genmats = []
#     unique_subspaces = []
#     for g in range(n_groups):
#         found = False
#         for uid, (ugm, usubspace) in enumerate(zip(unique_genmats, unique_subspaces)):
#             if genmats_equal(list_genmat[g], ugm) and list_start_end[g] == usubspace:
#                 genmat_group_id.append(uid)
#                 found = True
#                 break
#         if not found:
#             genmat_group_id.append(len(unique_genmats))
#             unique_genmats.append(list_genmat[g])
#             unique_subspaces.append(list_start_end[g])
#
#     unique_sizes = []
#     unique_offsets = [0]
#     seen_ids = []
#     for g in range(n_groups):
#         uid = genmat_group_id[g]
#         if uid not in seen_ids:
#             seen_ids.append(uid)
#             sz = len(initial_theta[g])
#             unique_sizes.append(sz)
#             unique_offsets.append(unique_offsets[-1] + sz)
#
#     n_total = sum(unique_sizes)
#     x0 = (np.concatenate([np.atleast_1d(np.asarray(initial_theta[g], dtype=float))
#                           for g in range(n_groups)
#                           if genmat_group_id[g] not in genmat_group_id[:g]])
#           if n_total > 0 else np.array([]))
#
#
#     full_basis = []
#     list_start_end = []
#     curr_ptr = 0
#     for g in range(n_groups):
#         states = list_ex_states[g]
#         full_basis.extend(states)
#         end_ptr = curr_ptr + len(states)
#         list_start_end.append((curr_ptr, end_ptr))
#         curr_ptr = end_ptr
#
#     H_full = np.array(construct_Hmat_CSFs_paral_triu(full_basis, Enuc, obt, tbt, nparal))
#
#     def unpack(xflat):
#         return [xflat[offsets[g]:offsets[g + 1]] for g in range(len(sizes))]
#
#     def unpack_unique(xflat):
#         return [xflat[unique_offsets[i]:unique_offsets[i + 1]]
#                 for i in range(len(unique_sizes))]
#
#     if list_theta_prev is not None:
#         clean_theta_prev = []
#         for th in list_theta_prev:
#             if isinstance(th, (float, np.floating)):
#                 tmp = [[] for _ in range(n_groups)]
#                 tmp[0] = [th]
#                 clean_theta_prev.append(tmp)
#             elif isinstance(th, np.ndarray):
#                 if len(th) == n_total:
#                     clean_theta_prev.append(unpack(th))
#                 else:
#                     if debug:
#                         print(f"Warning: Skipping incompatible theta_prev with length {len(th)} (expected {n_total})")
#                     continue
#             elif isinstance(th, list):
#                 if len(th) == n_groups:
#                     clean_theta_prev.append(th)
#                 else:
#                     if debug:
#                         print(f"Warning: Skipping theta_prev with {len(th)} groups (expected {n_groups})")
#                     continue
#             else:
#                 clean_theta_prev.append(th)
#
#         list_theta_prev = clean_theta_prev
#
#     def build_all_Uvecs(theta):
#         all_Uvecs = []
#         list_Umat_per_group = []
#
#         theta_grouped = theta
#         list_theta_prev_grouped = list_theta_prev
#
#         for g in range(n_groups):
#             ndim = len(list_ex_states[g])
#             Umat_total = np.eye(ndim)
#
#             # Compose with previous trains
#             for theta_prev_g in list_theta_prev_grouped:
#                 if g >= len(theta_prev_g) or len(theta_prev_g[g]) == 0:
#                     continue
#
#                 if len(list_genmat[g]) == 0:
#                     if debug:
#                         print(f"Warning: Group {g} has no generators, skipping theta_prev application")
#                     continue
#
#                 if debug:
#                     print(
#                         f"Group {g}: ndim={ndim}, len(theta_prev)={len(theta_prev_g[g])}, len(genmat)={len(list_genmat[g])}")
#
#                 Umat_p = make_Umat_decomp_genmat(
#                     list_genmat[g],
#                     theta_prev_g[g]
#                 )
#
#                 if debug:
#                     print(f"Group {g}: ndim={ndim}, len(theta)={len(theta_grouped[g])}")
#                 if hasattr(Umat_p, "toarray"):
#                     Umat_p = Umat_p.toarray()
#
#                 Umat_total = Umat_p @ Umat_total
#
#             # Compose with current train
#             if len(theta_grouped[g]) != 0:
#                 Umat_curr = make_Umat_decomp_genmat(
#                     list_genmat[g],
#                     theta_grouped[g]
#                 )
#                 if hasattr(Umat_curr, "toarray"):
#                     Umat_curr = Umat_curr.toarray()
#
#                 Umat_total = Umat_curr @ Umat_total
#             else:
#                 Umat_curr = np.eye(ndim)
#
#             # Unfold marked configurations
#             active_indices = np.where(list_sym_CSF_vec[g] != 0.0)[0]
#             for idx in active_indices:
#                 inivec = np.zeros(ndim)
#                 inivec[idx] = 1.0
#                 all_Uvecs.append((g, Umat_total @ inivec))
#
#             # Fixed extra basis vectors — theta-independent, added here
#             for extra_vec in list_list_extra_vecs[g]:
#                 all_Uvecs.append((g, extra_vec))
#
#             list_Umat_per_group.append(Umat_curr)
#
#         return all_Uvecs, list_Umat_per_group
#
#     def build_projected_matrices(theta):
#         t0 = time.time()
#         all_Uvecs, _ = build_all_Uvecs(theta)
#         t1 = time.time()
#
#         H_eff = np.zeros((n_cols, n_cols))
#         S_eff = np.zeros((n_cols, n_cols))
#
#
#         for i in range(n_cols):
#             gi, ui = all_Uvecs[i]
#             si, ei = list_start_end[gi]
#             for j in range(i, n_cols):
#                 gj, uj = all_Uvecs[j]
#                 sj, ej = list_start_end[gj]
#                 H_block = H_full[si:ei, sj:ej]
#                 Hij = ui @ (H_block @ uj)
#                 Sij = ui @ uj if gi == gj else 0.0
#                 H_eff[i, j] = H_eff[j, i] = Hij
#                 S_eff[i, j] = S_eff[j, i] = Sij
#         t2 = time.time()
#
#         return H_eff, S_eff
#
#     def f(xflat):
#         t0 = time.time()
#
#         theta = unpack(xflat)
#         t1 = time.time()
#
#         H_eff, S_eff = build_projected_matrices(theta)
#         t2 = time.time()
#
#         evals = scipy.linalg.eigh(H_eff, S_eff, eigvals_only=True)
#         evals = np.sort(evals)[:no_evals]
#         t3 = time.time()
#
#         result = float(np.dot(weight, evals))
#         return result
#
#     print("DEBUG INFO:")
#     print(f"sizes = {sizes}")
#     print(f"offsets = {offsets}")
#     print(f"n_total = {n_total}")
#     print(f"x0 = {x0}")
#     print(f"x0 shape = {x0.shape}")
#
#     if n_total == 0 or method is None:
#         x_opt = x0
#     else:
#         res = minimize(f, x0, method = method,tol=conv_threshold,options={'maxiter': 5000})
#         x_opt = res.x
#
#     # theta_opt = unpack(x_opt)
#
#     unique_thetas_opt = unpack_unique(x_opt)
#     theta_opt = [unique_thetas_opt[genmat_group_id[g]].tolist() for g in range(n_groups)]
#
#     all_Uvecs_opt, list_Umat_opt = build_all_Uvecs(theta_opt)
#
#     UCSF_opt = []
#     final_Uvec_list = []
#     for i in range(n_cols):
#         g_idx, u_vec = all_Uvecs_opt[i]
#         UCSF = make_UCSF_state(list_ex_states[g_idx], u_vec)
#         UCSF_opt.append(UCSF)
#         final_Uvec_list.append(u_vec)
#
#     return theta_opt, final_Uvec_list, list_Umat_opt, UCSF_opt

# Theta-gradient controls:
#   _USE_ANALYTIC_THETA_GRAD  — use the analytic (Hellmann-Feynman) gradient
#     instead of the parallel finite-difference one.
#   _CHECK_ANALYTIC_THETA_GRAD — on the FIRST gradient, also compute the finite
#     difference and assert the two agree (verifies on real data before relying
#     on the analytic version). Set False once trusted to skip the one-time cost.
_USE_ANALYTIC_THETA_GRAD = True
_CHECK_ANALYTIC_THETA_GRAD = False

def opt_theta_for_energy_UCSF_group(list_ex_states, initial_theta, list_genmat, l_use_decomp_genmat,
                                    Enuc, obt, tbt, nparal, method, list_sym_CSF_vec, no_evals, weight,
                                    shared_unitary_groups=None, list_theta_prev=None,
                                    list_list_extra_vecs=None, conv_threshold=1e-3, debug=False,
                                    frozen_theta_map=None):

    if list_list_extra_vecs is None:
        list_list_extra_vecs = [[] for _ in range(n_groups)]

    if list_theta_prev is None:
        list_theta_prev = []

    # frozen_theta_map: {(group_idx, theta_pos): fixed_value}
    # Those positions are held at their fixed values regardless of what the optimizer does.
    # Theory: when an excitation is degenerate with the reference (ΔE=0), the 2-state
    # problem tan(2θ)=2H/(E_ref-E_exc)→∞ gives θ=π/4 exactly.  No optimization needed.
    if frozen_theta_map is None:
        frozen_theta_map = {}

    n_groups = len(list_ex_states)
    # Each group contributes exactly ONE UCSF column: the sym_CSF_vec itself is the
    # starting vector (already encodes symmetric/antisymmetric combination).
    # Using separate unit vectors for each non-zero element of sym_CSF_vec would create
    # independent basis states whose mixing is determined by the eigenvector of the UCSF
    # Hamiltonian — any tiny H asymmetry from orbital optimization would break symmetry.
    n_cols = sum([1 if np.any(mask != 0.0) else 0 for mask in list_sym_CSF_vec]) \
             + sum([len(vecs) for vecs in list_list_extra_vecs])

    # Build shared theta mapping from shared_unitary_groups
    if shared_unitary_groups is not None:
        group_to_parent = {}
        for parent, members in shared_unitary_groups.items():
            for g in members:
                group_to_parent[g] = parent
    else:
        group_to_parent = {g: g for g in range(n_groups)}

    # Build x0 from unique parents only
    seen_parents = []
    unique_sizes = []
    unique_offsets = [0]
    for g in range(n_groups):
        parent = group_to_parent[g]
        if parent not in seen_parents:
            seen_parents.append(parent)
            sz = len(initial_theta[g])
            unique_sizes.append(sz)
            unique_offsets.append(unique_offsets[-1] + sz)

    n_total = sum(unique_sizes)
    x0 = (np.concatenate([np.atleast_1d(np.asarray(initial_theta[g], dtype=float))
                           for g in range(n_groups)
                           if group_to_parent[g] == g])
          if n_total > 0 else np.array([]))

    # Pre-set frozen positions in x0 to their fixed values so the optimizer
    # starts at the correct point and the zero-gradient condition holds.
    parent_to_uid_x0 = {parent: uid for uid, parent in enumerate(seen_parents)}
    for (g_frz, pos_frz), val_frz in frozen_theta_map.items():
        parent_frz = group_to_parent.get(g_frz, g_frz)
        if parent_frz in parent_to_uid_x0:
            uid_frz = parent_to_uid_x0[parent_frz]
            flat_pos = unique_offsets[uid_frz] + pos_frz
            if flat_pos < len(x0):
                x0[flat_pos] = val_frz

    print('unique_offsets:', unique_offsets)
    print(x0)

    full_basis = []
    list_start_end = []
    curr_ptr = 0
    for g in range(n_groups):
        states = list_ex_states[g]
        full_basis.extend(states)
        end_ptr = curr_ptr + len(states)
        list_start_end.append((curr_ptr, end_ptr))
        curr_ptr = end_ptr

    H_full = np.array(construct_Hmat_CSFs_paral_triu(full_basis, Enuc, obt, tbt, nparal))

    # True Gram matrix G[k,l] = <full_basis[k] | full_basis[l]> over the concatenated
    # basis of all groups — INCLUDING within-group overlaps.  Chain/basis-extension
    # states are appended raw (no orthogonalization), so states within a group may
    # legitimately overlap; the projected S matrix accounts for it via this Gram.
    total_dim = curr_ptr
    G = np.eye(total_dim)
    for gi in range(n_groups):
        si, ei = list_start_end[gi]
        for gj in range(gi, n_groups):
            sj, ej = list_start_end[gj]
            for k in range(ei - si):
                l_start = k + 1 if gi == gj else 0
                for l in range(l_start, ej - sj):
                    ov = overlap_CSFs(full_basis[si + k], full_basis[sj + l])
                    if abs(ov) > 1e-12:
                        G[si + k, sj + l] = ov
                        G[sj + l, si + k] = ov

    def unpack_unique(xflat):
        return [xflat[unique_offsets[i]:unique_offsets[i+1]]
                for i in range(len(unique_sizes))]

    def unpack(xflat):
        unique_thetas = unpack_unique(xflat)
        parent_to_uid = {parent: uid for uid, parent in enumerate(seen_parents)}
        return [unique_thetas[parent_to_uid[group_to_parent[g]]]
                for g in range(n_groups)]

    x0_full = unpack(x0)
    print("Full expanded thetas (all groups):")
    for g in range(n_groups):
        print(f"  Group {g} (parent={group_to_parent[g]}): {x0_full[g]}")

    if list_theta_prev is not None:
        clean_theta_prev = []
        for th in list_theta_prev:
            if isinstance(th, (float, np.floating)):
                tmp = [[] for _ in range(n_groups)]
                tmp[0] = [th]
                clean_theta_prev.append(tmp)
            elif isinstance(th, np.ndarray):
                if len(th) == n_total:
                    clean_theta_prev.append(unpack(th))
                else:
                    if debug:
                        print(f"Warning: Skipping incompatible theta_prev with length {len(th)} (expected {n_total})")
                    continue
            elif isinstance(th, list):
                if len(th) == n_groups:
                    clean_theta_prev.append(th)
                else:
                    if debug:
                        print(f"Warning: Skipping theta_prev with {len(th)} groups (expected {n_groups})")
                    continue
            else:
                clean_theta_prev.append(th)

        list_theta_prev = clean_theta_prev

    def build_all_Uvecs(theta):
        # Override frozen positions with their fixed values regardless of
        # what the optimizer put there.  This hard-freezes ref-degenerate
        # pair amplitudes at π/4 (determined by symmetry, not variational).
        theta_eff = [np.array(t, dtype=float) for t in theta]
        for (g_frz, pos_frz), val_frz in frozen_theta_map.items():
            if 0 <= g_frz < len(theta_eff) and 0 <= pos_frz < len(theta_eff[g_frz]):
                theta_eff[g_frz][pos_frz] = val_frz

        all_Uvecs = []
        list_Umat_per_group = []

        theta_grouped = theta_eff
        list_theta_prev_grouped = list_theta_prev

        for g in range(n_groups):
            ndim = len(list_ex_states[g])
            Umat_total = np.eye(ndim)

            for theta_prev_g in list_theta_prev_grouped:
                if g >= len(theta_prev_g) or len(theta_prev_g[g]) == 0:
                    continue
                if len(list_genmat[g]) == 0:
                    if debug:
                        print(f"Warning: Group {g} has no generators, skipping theta_prev application")
                    continue
                if debug:
                    print(f"Group {g}: ndim={ndim}, len(theta_prev)={len(theta_prev_g[g])}, len(genmat)={len(list_genmat[g])}")

                Umat_p = make_Umat_decomp_genmat(list_genmat[g], theta_prev_g[g])
                if hasattr(Umat_p, "toarray"):
                    Umat_p = Umat_p.toarray()
                Umat_total = Umat_p @ Umat_total

            if len(theta_grouped[g]) != 0:
                Umat_curr = make_Umat_decomp_genmat(list_genmat[g], theta_grouped[g])
                if hasattr(Umat_curr, "toarray"):
                    Umat_curr = Umat_curr.toarray()
                Umat_total = Umat_curr @ Umat_total
            else:
                Umat_curr = np.eye(ndim)

            # Use sym_CSF_vec directly as the single starting vector.
            # For a non-degenerate group: sym_CSF_vec = [1,0,...] → same as before.
            # For a combined degenerate group: sym_CSF_vec = [1/√2,...,±1/√2,...]
            # → rotation starts from the pre-combined symmetric state, giving ONE
            # UCSF column. This prevents H asymmetry from the orbital optimisation
            # breaking the degeneracy through the eigenvector mixing step.
            inivec = list_sym_CSF_vec[g].copy()
            norm = np.linalg.norm(inivec)
            if norm > 1e-10:
                all_Uvecs.append((g, Umat_total @ inivec))

            # Extra columns (chain/basis-extension states) share the group's
            # unitary exactly: same W applied to a different starting vector.
            # Since W is orthogonal and the starting vectors are mutually
            # orthogonal, the resulting UCSF columns stay orthogonal.
            for extra_vec in list_list_extra_vecs[g]:
                all_Uvecs.append((g, Umat_total @ extra_vec))

            list_Umat_per_group.append(Umat_curr)

        return all_Uvecs, list_Umat_per_group

    def build_projected_matrices(theta):
        all_Uvecs, _ = build_all_Uvecs(theta)

        # Vectorized projection. Each U-vector ui lives only in its group's rows
        # [si:ei] of the full space, so embedding every ui as a column of V
        # (nonzero only in those rows) makes the whole n_cols^2 double loop equal
        # to two matrix products:
        #     H_eff = V^T H_full V ,  S_eff = V^T G V
        # This replaces ~n_cols^2 GIL-held Python iterations with two BLAS matmuls,
        # so f() is entirely GIL-releasing and the parallel gradient scales.
        _ndim_full = H_full.shape[0]
        V = np.zeros((_ndim_full, n_cols))
        for _i, (_gi, _ui) in enumerate(all_Uvecs):
            _si, _ei = list_start_end[_gi]
            V[_si:_ei, _i] = _ui
        H_eff = V.T @ H_full @ V
        S_eff = V.T @ G @ V
        # Symmetrize to clean up floating-point asymmetry (H_full and G are
        # symmetric, so V^T H_full V is symmetric up to rounding).
        H_eff = 0.5 * (H_eff + H_eff.T)
        S_eff = 0.5 * (S_eff + S_eff.T)

        if _CHECK_VECTORIZED_H:
            # One-off gate: reproduce H_eff/S_eff with the original per-pair loop
            # and assert equality. Flip _CHECK_VECTORIZED_H off once verified.
            H_ref = np.zeros((n_cols, n_cols))
            S_ref = np.zeros((n_cols, n_cols))
            for i in range(n_cols):
                gi, ui = all_Uvecs[i]
                si, ei = list_start_end[gi]
                for j in range(i, n_cols):
                    gj, uj = all_Uvecs[j]
                    sj, ej = list_start_end[gj]
                    Hij = ui @ (H_full[si:ei, sj:ej] @ uj)
                    Sij = ui @ (G[si:ei, sj:ej] @ uj)
                    H_ref[i, j] = H_ref[j, i] = Hij
                    S_ref[i, j] = S_ref[j, i] = Sij
            assert np.allclose(H_eff, H_ref, atol=1e-10), \
                f'vectorized H_eff mismatch: {np.max(np.abs(H_eff - H_ref)):.3e}'
            assert np.allclose(S_eff, S_ref, atol=1e-10), \
                f'vectorized S_eff mismatch: {np.max(np.abs(S_eff - S_ref)):.3e}'

        # S need NOT be the identity: distinct reference CSFs (mu != nu) generate
        # excitation manifolds that legitimately overlap, since
        #   <UCSF_mu|UCSF_nu> = <CSF_mu| W_mu^dag W_nu |CSF_nu>
        # is generally non-zero.  The generalized eigenproblem H c = eps S c
        # (scipy.linalg.eigh(H_eff, S_eff)) handles this exactly.
        # What WOULD break the solve is a singular / linearly-dependent basis,
        # i.e. S losing positive-definiteness.  Guard against THAT, not non-identity.
        S_cond_tol = 1e-8  # smallest allowed eigenvalue of S
        s_evals = np.linalg.eigvalsh(S_eff)
        min_s_eval = s_evals[0]
        if min_s_eval < S_cond_tol:
            print('ERROR: UCSF overlap matrix S is (near-)singular — basis is '
                  'linearly dependent.')
            print(f'  Smallest eigenvalue of S = {min_s_eval:.3e} '
                  f'(tol = {S_cond_tol:.1e})')
            # Report the most non-orthogonal column pairs to locate the redundancy
            offdiag = np.abs(S_eff - np.diag(np.diag(S_eff)))
            flat = np.dstack(np.unravel_index(np.argsort(offdiag.ravel())[::-1],
                                              offdiag.shape))[0]
            print('  Largest off-diagonal overlaps (likely redundant columns):')
            shown = 0
            for (i_bad, j_bad) in flat:
                if i_bad >= j_bad:
                    continue
                if offdiag[i_bad, j_bad] < 1e-6:
                    break
                gi_b, ui_b = all_Uvecs[i_bad]
                gj_b, uj_b = all_Uvecs[j_bad]
                si_b, ei_b = list_start_end[gi_b]
                sj_b, ej_b = list_start_end[gj_b]
                print(f'    S[{i_bad},{j_bad}] = {S_eff[i_bad, j_bad]:+.6e}  '
                      f'(group {gi_b} vs {gj_b})')
                Gblock = G[si_b:ei_b, sj_b:ej_b]
                for (k, l) in np.argwhere(np.abs(Gblock) > 1e-12)[:5]:
                    if abs(ui_b[k] * uj_b[l]) < 1e-9:
                        continue
                    print(f'      shared CSF: gi[{k}] == gj[{l}], '
                          f'ui={ui_b[k]:+.4f}, uj={uj_b[l]:+.4f}, '
                          f'SD={list_ex_states[gi_b][k][0][0]}')
                shown += 1
                if shown >= 10:
                    break
            print('\n  === Group generator summary (for overlap diagnosis) ===')
            for _dg in range(n_groups):
                _ref = list_ex_states[_dg][0]
                _ref_idx = sorted(int(x) for x in _ref[1])
                _ref_coef = [round(float(c), 4) for c in _ref[2]]
                _n_states = len(list_ex_states[_dg])
                _n_gen = len(list_genmat[_dg]) if _dg < len(list_genmat) else 0
                print(f'  Group {_dg}: n_states={_n_states}  n_generators={_n_gen}')
                print(f'    ref SD-idx={_ref_idx}  coefs={_ref_coef}')
                for _k, _st in enumerate(list_ex_states[_dg]):
                    _sidx = sorted(int(x) for x in _st[1])
                    _sc   = [round(float(c), 4) for c in _st[2]]
                    print(f'    slot {_k}: SD-idx={_sidx}  coefs={_sc}')
            # Highlight which groups are fully identical (S≈1)
            _bad_pairs = set()
            for (i_bad, j_bad) in flat:
                if i_bad >= j_bad: continue
                if offdiag[i_bad, j_bad] < 1e-6: break
                gi_b = all_Uvecs[i_bad][0]
                gj_b = all_Uvecs[j_bad][0]
                if abs(S_eff[i_bad, j_bad]) > 0.99:
                    _bad_pairs.add((gi_b, gj_b))
            for (_ga, _gb) in _bad_pairs:
                print(f'\n  Groups {_ga} and {_gb} are near-identical (S≈1):')
                _ra = [sorted(int(x) for x in st[1]) for st in list_ex_states[_ga]]
                _rb = [sorted(int(x) for x in st[1]) for st in list_ex_states[_gb]]
                _same = [idx for idx in _ra if idx in _rb]
                print(f'    shared SD-idx sets: {_same}')
            print('  Bombing out!')
            sys.exit()

        return H_eff, S_eff

    def f(xflat):
        theta = unpack(xflat)
        H_eff, S_eff = build_projected_matrices(theta)
        evals = scipy.linalg.eigh(H_eff, S_eff, eigvals_only=True)
        evals = np.sort(evals)[:no_evals]
        return float(np.dot(weight, evals))

    def _theta_grad(xflat):
        """Parallel central finite-difference gradient of f.

        f is a pure, numpy-heavy function of theta (matrix projection +
        generalized eigensolve — both release the GIL), so the components are
        spread across threads: real parallelism over the theta parameters with
        no pickling of H_full/G to workers. This is what lets a gradient method
        actually use nparal cores, unlike derivative-free Powell.
        """
        _tg0 = time.time()
        xflat = np.asarray(xflat, dtype=float)
        n = len(xflat)
        eps = 1.0e-6
        def _one(k):
            xp = xflat.copy(); xp[k] += eps
            xm = xflat.copy(); xm[k] -= eps
            return (f(xp) - f(xm)) / (2.0 * eps)
        if nparal > 1 and n > 1:
            njobs = min(nparal, n)
            def _chunk(chunk):
                return [_one(int(k)) for k in chunk]
            parts = Parallel(n_jobs=njobs, prefer='threads')(
                delayed(_chunk)(ch) for ch in np.array_split(np.arange(n), njobs))
            g = np.array([v for part in parts for v in part], dtype=float)
        else:
            g = np.array([_one(k) for k in range(n)], dtype=float)
        _tg1 = time.time()
        # Diagnostic: per-gradient wall time + gradient norm. Tells us whether the
        # optimizer is slow because each gradient is expensive (thread scaling /
        # per-eval cost) vs because L-BFGS is doing many iterations.
        print(f'[theta-grad] {n} params, {2*n+1} evals across {min(nparal, n)} workers '
              f'took {_tg1 - _tg0:.2f}s  |grad|={np.linalg.norm(g):.3e}')
        return g

    def _theta_grad_analytic(xflat):
        """Analytic gradient via the generalized Hellmann-Feynman theorem:
            df/dtheta_j = 2 * sum_k w_k * (dpsi_k/dtheta_j) . rho_k
        with psi_k = V c_k and rho_k = H_full psi_k - E_k G psi_k, from ONE
        eigensolve. Each dpsi_k/dtheta_j touches only the column(s) of the group
        owning theta_j. Requires E_k non-degenerate — the oracle check in the
        dispatch below verifies this against finite difference on real data.
        """
        xflat = np.asarray(xflat, dtype=float)
        theta_grouped = unpack(xflat)
        theta_eff = [np.array(t, dtype=float) for t in theta_grouped]
        for (g_frz, pos_frz), val_frz in frozen_theta_map.items():
            if 0 <= g_frz < len(theta_eff) and 0 <= pos_frz < len(theta_eff[g_frz]):
                theta_eff[g_frz][pos_frz] = val_frz

        parent_to_uid = {parent: uid for uid, parent in enumerate(seen_parents)}
        ndim_full = H_full.shape[0]

        def _cols_and_derivs(g, theta_g, start_vecs):
            """Group g: list over start_vecs of (col, {ii: dcol/dtheta_{g,ii}}).
            col = U_m..U_1 v ; dcol/dtheta_ii = U_m..U_{ii+1} (dU_ii) U_{ii-1}..U_1 v."""
            gm_list = list_genmat[g]
            m = len(gm_list)
            U, dU = [], []
            for ii in range(m):
                uev, prj, evec, se = gm_list[ii]
                Uii  = make_analytical_U_decomp_genmat(theta_g[ii], uev, prj, evec, se)
                dUii = make_analytical_dU_decomp_genmat(theta_g[ii], uev, prj, evec, se)
                U.append(Uii.toarray()  if hasattr(Uii,  'toarray') else np.asarray(Uii))
                dU.append(dUii.toarray() if hasattr(dUii, 'toarray') else np.asarray(dUii))
            out = []
            for v in start_vecs:
                w = [np.asarray(v, dtype=float)]
                for ii in range(m):
                    w.append(U[ii] @ w[-1])          # w[ii+1] = U_ii..U_1 v
                col = w[m]
                derivs = {}
                for ii in range(m):
                    temp = dU[ii] @ w[ii]            # dU_ii @ (U_{ii-1}..U_1 v)
                    for kk in range(ii + 1, m):
                        temp = U[kk] @ temp           # left-apply U_{ii+1}..U_m
                    derivs[ii] = temp
                out.append((col, derivs))
            return out

        V = np.zeros((ndim_full, n_cols))
        col_records = []   # (col_index, group g, {ii: dcol_block_vec})
        ci = 0
        for g in range(n_groups):
            s, e = list_start_end[g]
            start_vecs = []
            inivec = np.asarray(list_sym_CSF_vec[g], dtype=float)
            if np.linalg.norm(inivec) > 1e-10:
                start_vecs.append(inivec)
            for extra in list_list_extra_vecs[g]:
                start_vecs.append(np.asarray(extra, dtype=float))
            if not start_vecs:
                continue
            for (col, derivs) in _cols_and_derivs(g, theta_eff[g], start_vecs):
                V[s:e, ci] = col
                col_records.append((ci, g, derivs))
                ci += 1
        assert ci == n_cols, f'column count mismatch: built {ci}, expected {n_cols}'

        H_eff = V.T @ H_full @ V
        S_eff = V.T @ G @ V
        H_eff = 0.5 * (H_eff + H_eff.T)
        S_eff = 0.5 * (S_eff + S_eff.T)
        evals, C = scipy.linalg.eigh(H_eff, S_eff)   # C S-normalized: C^T S_eff C = I
        order = np.argsort(evals)[:no_evals]

        state_c, state_rho = [], []
        for k in order:
            c = C[:, k]
            psi = V @ c
            state_c.append(c)
            state_rho.append(H_full @ psi - evals[k] * (G @ psi))

        g_out = np.zeros(len(xflat))
        for (ci_, g, derivs) in col_records:
            uid = parent_to_uid[group_to_parent[g]]
            s, e = list_start_end[g]
            for ii, dcol in derivs.items():
                flat_j = unique_offsets[uid] + ii
                acc = 0.0
                for kk in range(len(order)):
                    acc += weight[kk] * state_c[kk][ci_] * float(dcol @ state_rho[kk][s:e])
                g_out[flat_j] += 2.0 * acc

        for (g_frz, pos_frz), val_frz in frozen_theta_map.items():
            parent = group_to_parent.get(g_frz, g_frz)
            if parent in parent_to_uid:
                fj = unique_offsets[parent_to_uid[parent]] + pos_frz
                if 0 <= fj < len(g_out):
                    g_out[fj] = 0.0
        return g_out

    _analytic_checked = [False]

    def _theta_grad_dispatch(xflat):
        if not _USE_ANALYTIC_THETA_GRAD:
            return _theta_grad(xflat)
        _ta0 = time.time()
        g_an = _theta_grad_analytic(xflat)
        _ta1 = time.time()
        if _CHECK_ANALYTIC_THETA_GRAD and not _analytic_checked[0]:
            # One-time verification on REAL data: compare against finite difference.
            g_fd = _theta_grad(xflat)
            err = np.max(np.abs(g_an - g_fd))
            denom = max(1.0, float(np.max(np.abs(g_fd))))
            print(f'[theta-grad CHECK] max|analytic - finite_diff| = {err:.3e} '
                  f'(rel {err/denom:.3e})')
            assert err < 1e-4 * denom + 1e-7, \
                f'analytic theta gradient disagrees with finite difference: {err:.3e}'
            print('[theta-grad CHECK] analytic gradient VERIFIED vs finite difference')
            _analytic_checked[0] = True
        print(f'[theta-grad] analytic ({len(xflat)} params) took {_ta1 - _ta0:.2f}s '
              f'|grad|={np.linalg.norm(g_an):.3e}')
        return g_an

    print("DEBUG INFO:")
    print(f"unique_sizes = {unique_sizes}")
    print(f"unique_offsets = {unique_offsets}")
    print(f"n_total = {n_total}")
    print(f"x0 = {x0}")
    print(f"x0 shape = {x0.shape}")
    print(f"group_to_parent = {group_to_parent}")
    print(f"seen_parents = {seen_parents}")

    # exp(theta*G) is 2π-periodic: keep all angles in [-π, π] so warm starts
    # cannot accumulate across rounds (e.g. Powell drifting to θ ≈ -29).
    x0 = (np.asarray(x0, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi

    if n_total == 0 or method is None:
        x_opt = x0
    else:
        E_before = float(f(x0))
        _grad_methods = ('BFGS', 'CG', 'L-BFGS-B', 'TNC', 'SLSQP', 'Newton-CG')
        # disp=True → scipy prints per-iteration progress so we can see whether
        # the optimizer is iterating (and how many times) rather than hung.
        _opts = _solver_options(method)
        _opts['disp'] = True
        if method in _grad_methods:
            # Gradient method + parallel finite-difference gradient: converges in
            # far fewer objective evaluations than derivative-free Powell at this
            # dimensionality, and the gradient spreads the theta components across
            # nparal cores (Powell can use neither).
            res = minimize(f, x0, method=method, jac=_theta_grad_dispatch,
                           tol=conv_threshold, options=_opts)
        else:
            res = minimize(f, x0, method=method, tol=conv_threshold,
                           options=_opts)
        x_opt = (np.asarray(res.x, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi
        E_after = float(res.fun)

        print(f'\n[theta-opt] E_before={E_before:.10f}  E_after={E_after:.10f}  '
              f'ΔE={E_after - E_before:+.3e}')
        if not res.success:
            print(f'  ⚠ optimizer did not converge: {res.message}  (nit={res.nit})')
        if E_after > E_before + 1e-10:
            print(f'  ⚠ energy INCREASED after optimization — likely stuck in wrong basin')
            print(f'    Reverting to x0')
            x_opt = x0

        # Shared-theta consistency check: all members of a shared group must have
        # identical thetas (they pull from the same parent slot in x_opt).
        if shared_unitary_groups is not None:
            theta_opt_check = unpack(x_opt)
            for parent, members in shared_unitary_groups.items():
                if len(members) < 2: continue
                t0 = np.array(theta_opt_check[members[0]])
                for m in members[1:]:
                    tm = np.array(theta_opt_check[m])
                    if not np.allclose(t0, tm, atol=1e-12):
                        print(f'  ⚠ shared-theta mismatch: group {members[0]} vs {m}: '
                              f'{t0} vs {tm}')

    # ── Grid-seeded restart to escape local basins in the theta landscape ─────
    # L-BFGS-B is a local optimizer: on the multimodal theta profiles that appear
    # in strongly-correlated (bond-breaking) geometries it can settle in a basin
    # that a coarse 1D grid scan sees past. Each round we grid-scan every
    # large-|θ| coordinate; any coordinate whose grid minimum beats the current
    # energy is reseeded, and we re-run the local optimizer from the reseeded
    # vector (the polish repairs the inter-coordinate coupling a single-coordinate
    # grid move ignores). We accept a round only if the energy actually drops, and
    # stop as soon as a round reseeds nothing or the polish fails to improve.
    _scan_threshold = 0.5   # rad — only scan coordinates with large angles
    _n_scan         = 25
    _scan_angles    = np.linspace(-np.pi, np.pi, _n_scan)
    _refine_rounds  = 3
    _accept_tol     = 1e-6  # Ha — ignore improvements below this

    def _wrap(_a):
        return (np.asarray(_a, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi

    if n_total > 0 and method is not None:
        _E_cur = float(f(x_opt))
        for _r in range(_refine_rounds):
            _x_seed = x_opt.copy()
            _n_reseeded = 0
            # Coordinates worth scanning this round (large angles only).
            _scan_coords = [i for i in range(len(x_opt))
                            if abs(x_opt[i]) >= _scan_threshold]
            if _scan_coords:
                # Build every (coord, angle) trial vector and evaluate f on all of
                # them at once. f is pure and GIL-releasing (matrix projection +
                # generalized eigensolve), so the evals spread across nparal threads
                # exactly like the gradient does — no serial grind on one core.
                _scan_vecs = []
                for _ci in _scan_coords:
                    for _th in _scan_angles:
                        _xv = x_opt.copy(); _xv[_ci] = _th
                        _scan_vecs.append(_xv)
                _ts0 = time.time()
                if nparal > 1 and len(_scan_vecs) > 1:
                    _scan_E = Parallel(n_jobs=min(nparal, len(_scan_vecs)),
                                       prefer='threads')(
                        delayed(f)(_xv) for _xv in _scan_vecs)
                else:
                    _scan_E = [f(_xv) for _xv in _scan_vecs]
                _scan_E = np.asarray(_scan_E, dtype=float).reshape(
                    len(_scan_coords), _n_scan)
                print(f'[refine r{_r}] scanned {len(_scan_coords)} coords x '
                      f'{_n_scan} angles = {len(_scan_vecs)} evals across '
                      f'{min(nparal, len(_scan_vecs))} workers in '
                      f'{time.time() - _ts0:.1f}s')
                for _idx, _flat_i in enumerate(_scan_coords):
                    _energies  = _scan_E[_idx]
                    _imin      = int(np.argmin(_energies))
                    _best_scan = _scan_angles[_imin]
                    _best_E    = float(_energies[_imin])
                    # periodic angular distance: θ and θ ± 2π are the same point
                    _dx_basin  = abs(((_best_scan - x_opt[_flat_i] + np.pi)
                                      % (2.0 * np.pi)) - np.pi)
                    if _best_E < _E_cur - _accept_tol and _dx_basin > 0.3:
                        _uid = next(i for i in range(len(unique_offsets) - 1)
                                    if unique_offsets[i] <= _flat_i < unique_offsets[i+1])
                        print(f'[refine r{_r}] reseed flat_param[{_flat_i}] '
                              f'(parent={seen_parents[_uid]}): θ {x_opt[_flat_i]:.4f} → '
                              f'{_best_scan:.4f}   E {_E_cur:.8f} → {_best_E:.8f}')
                        _x_seed[_flat_i] = _best_scan
                        _n_reseeded += 1
            if _n_reseeded == 0:
                print(f'[refine r{_r}] no coordinate improved — refinement converged')
                break
            # polish from the reseeded vector (repairs inter-coordinate coupling)
            if method in _grad_methods:
                _res_ref = minimize(f, _x_seed, method=method,
                                    jac=_theta_grad_dispatch, tol=conv_threshold,
                                    options=_opts)
            else:
                _res_ref = minimize(f, _x_seed, method=method, tol=conv_threshold,
                                    options=_opts)
            _x_ref = _wrap(_res_ref.x)
            _E_ref = float(f(_x_ref))
            # keep the raw reseeded vector if the polish drifted back uphill
            _E_seed = float(f(_x_seed))
            if _E_seed < _E_ref:
                _x_ref, _E_ref = _x_seed, _E_seed
            if _E_ref < _E_cur - _accept_tol:
                print(f'[refine r{_r}] accepted: E {_E_cur:.8f} → {_E_ref:.8f} '
                      f'(Δ {_E_ref - _E_cur:+.2e}, reseeded {_n_reseeded})')
                x_opt, _E_cur = _x_ref, _E_ref
            else:
                print(f'[refine r{_r}] polish did not beat current E by tol — stopping')
                break
    # ── End grid-seeded restart ───────────────────────────────────────────────

    # Reconstruct per-group thetas (shared groups get identical values) from the
    # final (possibly refined) x_opt.
    parent_to_uid = {parent: uid for uid, parent in enumerate(seen_parents)}
    unique_thetas_opt = unpack_unique(x_opt)
    theta_opt = [unique_thetas_opt[parent_to_uid[group_to_parent[g]]].tolist()
                 for g in range(n_groups)]

    all_Uvecs_opt, list_Umat_opt = build_all_Uvecs(theta_opt)

    UCSF_opt = []
    final_Uvec_list = []
    for i in range(n_cols):
        g_idx, u_vec = all_Uvecs_opt[i]
        UCSF = make_UCSF_state(list_ex_states[g_idx], u_vec)
        UCSF_opt.append(UCSF)
        final_Uvec_list.append(u_vec)

    return theta_opt, final_Uvec_list, list_Umat_opt, UCSF_opt

def opt_one_UCSF(UCSF_state,list_doci_ex_space,list_genmat,group_mp2_ampld,U_vec,Enuc,obt,tbt,debug=False):
    """
    Given a UCSF state generated by make_UCSF_state, use the symmetry-adapted basis in this state
    to construct a Hamiltonian matrix and get the ground state within this space.
    """

    if debug: print('\nIn opt_one_UCSF')

   #[onlist,onidx_list,coefvec] = UCSF_state
   #basis_list = [list_doci_ex_space[0]]
   #for ibas, basis in enumerate(list_doci_ex_space[1:]):
   #    Selm = overlap_LCSD(UCSF_state[0],UCSF_state[1],UCSF_state[2],basis[0],basis[1],basis[2])
   #    print(ibas,Selm,U_vec[ibas+1])

    ndim = len(list_doci_ex_space)

    Hmat = np.zeros([ndim,ndim])
    for ibra, bas_bra in enumerate(list_doci_ex_space):
        for jket in range(ibra,len(list_doci_ex_space)):
            bas_ket = list_doci_ex_space[jket]
            Helm = Helm_between_LCSDs(Enuc,obt,tbt,bas_bra[0],bas_bra[2],bas_ket[0],bas_ket[2])
            Hmat[ibra,jket] = Helm
            Hmat[jket,ibra] = Helm

    E_GS, psi_GS = get_ground_state(Hmat)
    E_UCSF = U_vec.transpose()@Hmat@U_vec
    Selm = np.dot(psi_GS,U_vec)
    if debug:
        print(f'E_GS in opt_one_UCSF: {E_GS}, vs E_UCSF: {E_UCSF}')
        print(f'Overlap between U state and psi_GS: {Selm}')

   #eigval,eigvec = np.linalg.eigh(Hmat)
   #print('\nEigensolutions of Hmat:')
   #print_eigen_solution(eigval,eigvec)

    inivec = np.zeros(ndim)
    inivec[0] = 1.0

    def cost(x):
        return eval_ref_UCSF_ovlp(x, list_genmat,psi_GS,inivec)

    x0 = copy.deepcopy(group_mp2_ampld)

    options = {'maxiter' : 10000,
        'disp'    : True,
        'gtol'    : 1.0e-4
    }

    sol = minimize(cost, x0, method='BFGS',options=options)

    Umat_opt, Uvec_opt = make_and_apply_U_matrix(list_genmat,sol.x)

    Selm = np.dot(psi_GS,Uvec_opt)
    E_optUCSF = Uvec_opt.transpose()@Hmat@Uvec_opt
    if debug:
        print(f'Overlap between U_opt state and psi_GS: {Selm}')
        print(f'Energy of U_opt state and psi_GS: {E_optUCSF}')

    small = 0.95
    if abs(Selm) < small:
        print(f'Overlap between U_opt state and psi_GS < {small}')
        print('Bombing out!')
        sys.exit()

    Ustate_opt = make_UCSF_state(list_doci_ex_space,Uvec_opt)

    x_list = sol.x.tolist()

    return Ustate_opt, x_list

def compare_num_anl_Umat(x,list_genmat,list_decomp_genmat,nparal,debug=False):
    """
    Compare Umat created by numerical exponentiating genmat and analytical formula
    using decompoised genmat
    """

    tic = time.perf_counter()
    Umat_anl = make_Umat_decomp_genmat(list_decomp_genmat,x)
    toc = time.perf_counter()
    time_Umat_anl = toc - tic
    tic = time.perf_counter()
    Umat_paral = make_Umat_decomp_genmat_joblib(list_decomp_genmat,x,nparal)
    toc = time.perf_counter()
    time_Umat_paral = toc - tic
    print(f'Ratio of paral and sequantial anal. Umat calculation: {time_Umat_paral / time_Umat_anl}')
    tic = time.perf_counter()
    Umat_num, _ = make_and_apply_U_matrix(list_genmat,x)
    toc = time.perf_counter()
    time_Umat_num = toc - tic
    print(f'Time for preparing anal. and num. Umat, and their ratio: {time_Umat_anl,time_Umat_num,time_Umat_anl/time_Umat_num}')
    Umat_num = csr_matrix(Umat_num)

    assert np.isclose(scipy.sparse.linalg.norm(Umat_anl - Umat_num),0.0)
    assert np.isclose(scipy.sparse.linalg.norm(Umat_anl - Umat_paral),0.0)

def grad_comp_ref_UCSF_ovlp(icomp,x,list_genmat,refvec,inivec,l_use_decomp_genmat,eps=1.0e-4):
    """
    Numerical gradient of the eval_ref_UCSF_ovlp function. Component 0 gives the function value.
    Components 1 to len(x) gives the gradient components.
    """

    if icomp == 0:
        grad_comp = eval_ref_UCSF_ovlp(x,list_genmat,refvec,inivec,l_use_decomp_genmat)
    else:
        x_disp = x.copy()
        x_disp[icomp-1] = x[icomp-1] + eps
        func_plus = eval_ref_UCSF_ovlp(x_disp,list_genmat,refvec,inivec,l_use_decomp_genmat)
        x_disp[icomp-1] = x[icomp-1] - eps
        func_minus = eval_ref_UCSF_ovlp(x_disp,list_genmat,refvec,inivec,l_use_decomp_genmat)
        grad_comp = (func_plus - func_minus) / (2.0 * eps)

    return grad_comp

def grad_ref_UCSF_ovlp(x,list_genmat,refvec,inivec,nparal,l_use_decomp_genmat):

    l_paral = False
    if nparal != 1: l_paral = True
    list_grad_comp = []
    if l_paral:
        print(f'parallel in grad_ref_UCSF_ovlp with {nparal} cores, {len(x)} grad. components')
       #Good Parallel. Comment out the lines between Good Parallel to End of good parallel and
       #comment on the list_grad_comp = Parallel(n_jobs=nparal)(delayed(grad_comp_ref_UCSF_ovlp) line to return
       #to the original code.
        n_comp = len(x)+1
        nscale = 1
        chunk_size = 0
        while chunk_size < 1:
            nchunk = nparal // nscale
            if nchunk < 1: nchunk = 1
            chunk_size = n_comp // nchunk
            nscale *= 2
        list_all_idx = []
        for ii in range(n_comp): list_all_idx.append(ii)
        chunks = []
        for ii in range(0,nchunk):
            chunks.append(list_all_idx[ii*chunk_size:(ii+1)*chunk_size])
        chunks[-1] += list_all_idx[nchunk*chunk_size:]
        def grad_chunk_comp_ref_UCSF_ovlp(list_idx,x,list_genmat,refvec,inivec,l_use_decomp_genmat):
            list_grad_comp_1chunk = []
            for icomp in list_idx:
               #print(f'Handling icomp: {icomp}')
                grad_comp = grad_comp_ref_UCSF_ovlp(icomp,x,list_genmat,refvec,inivec,l_use_decomp_genmat)
                list_grad_comp_1chunk.append(grad_comp)

            return list_grad_comp_1chunk

        print(f'Before parallel in chunks, nchunks: {nchunk}')
        tic = time.perf_counter()
        list_list_grad_comp = Parallel(n_jobs=nchunk,prefer="threads")(delayed(grad_chunk_comp_ref_UCSF_ovlp)(chunk,x,list_genmat,refvec,inivec,l_use_decomp_genmat) for chunk in chunks)
        toc = time.perf_counter()
        print(f'Done parallel in chunks, used time: {toc - tic}')
        list_grad_comp = []
        for item in list_list_grad_comp: list_grad_comp += item
       #End of good parallel
       #from joblib import Parallel, delayed
       #list_grad_comp = Parallel(n_jobs=nparal)(delayed(grad_comp_ref_UCSF_ovlp)(ii,x,list_genmat,refvec,inivec,l_use_decomp_genmat) for ii in range(len(x)+1))
    else:
        print(f'serial in grad_ref_UCSF_ovlp with {nparal} cores, {len(x)} grad. components')
        tic = time.perf_counter()
        for icomp in range(len(x)+1):
            print(f'Looping over gradient components: {icomp}')
            grad_comp = grad_comp_ref_UCSF_ovlp(icomp,x,list_genmat,refvec,inivec,l_use_decomp_genmat)
            list_grad_comp.append(grad_comp)
        toc = time.perf_counter()
        print(f'Done serial in grad_ref_UCSF_ovlp, used time: {toc - tic}')

    funval = list_grad_comp[0]
    grad = np.array(list_grad_comp[1:])

    return funval,grad

def grad_ref_UCSF_ovlp_no_data_share(x,list_genmat,refvec,inivec,nparal,l_use_decomp_genmat):

    l_paral = False
    if nparal != 1: l_paral = True
    list_grad_comp = []
    if l_paral:
       #Good Parallel. Comment out the lines between Good Parallel to End of good parallel and
       #comment on the list_grad_comp = Parallel(n_jobs=nparal)(delayed(grad_comp_ref_UCSF_ovlp) line to return
       #to the original code.
        n_comp = len(x)+1
        chunks = even_distrib_ncomp_to_nparal(n_comp,nparal)
        nchunk = len(chunks)
       #chunk_size = 1
       #while(chunk_size*nparal < n_comp):
       #    chunk_size += 1

       #nchunk = n_comp // chunk_size
       #print(f'chunk_size: {chunk_size}, nchunk: {nchunk}, n_comp: {n_comp}, leftover: {n_comp-nchunk*chunk_size}')
       #assert n_comp-nchunk*chunk_size >= 0
       #list_all_idx = []
       #for ii in range(n_comp): list_all_idx.append(ii)
       #chunks = []
       #for ii in range(0,nchunk):
       #    chunks.append(list_all_idx[ii*chunk_size:(ii+1)*chunk_size])
       #
       #if len(list_all_idx[nchunk*chunk_size:]) != 0: chunks.append(list_all_idx[nchunk*chunk_size:])
       #nchunk = len(chunks)
       #print(f'final nchunk: {nchunk}')
       #for chunk in chunks: print(chunk)
        def grad_chunk_comp_ref_UCSF_ovlp(list_idx,x,list_genmat,refvec,inivec,l_use_decomp_genmat):
            list_grad_comp_1chunk = []
            for icomp in list_idx:
               #print(f'Handling icomp: {icomp}')
                grad_comp = grad_comp_ref_UCSF_ovlp(icomp,x,list_genmat,refvec,inivec,l_use_decomp_genmat)
                list_grad_comp_1chunk.append(grad_comp)

            return list_grad_comp_1chunk

        list_list_genmat = []
        list_refvec = []
        list_inivec = []
        list_l_use_decomp_genmat = []
        list_x = []
        for ichunk in range(nchunk):
            list_list_genmat.append(list_genmat)
            list_refvec.append(refvec)
            list_inivec.append(inivec)
            list_l_use_decomp_genmat.append(l_use_decomp_genmat)
            list_x.append(x)
       #print(f'Before parallel in chunks, # of chunks: {nchunk}')
        tic = time.perf_counter()
        list_list_grad_comp = Parallel(n_jobs=nchunk)(delayed(grad_chunk_comp_ref_UCSF_ovlp)(chunks[ii],list_x[ii],list_list_genmat[ii],list_refvec[ii],list_inivec[ii],list_l_use_decomp_genmat[ii]) for ii in range(nchunk))
        toc = time.perf_counter()
        print(f'Done parallel in chunks, used time: {toc - tic}')
        list_grad_comp = []
        for item in list_list_grad_comp: list_grad_comp += item
       #End of good parallel
       #from joblib import Parallel, delayed
       #list_grad_comp = Parallel(n_jobs=nparal)(delayed(grad_comp_ref_UCSF_ovlp)(ii,x,list_genmat,refvec,inivec,l_use_decomp_genmat) for ii in range(len(x)+1))
    else:
        print(f'serial in grad_ref_UCSF_ovlp with {nparal} cores, {len(x)} grad. components')
        tic = time.perf_counter()
        for icomp in range(len(x)+1):
            print(f'Looping over gradient components: {icomp}')
            grad_comp = grad_comp_ref_UCSF_ovlp(icomp,x,list_genmat,refvec,inivec,l_use_decomp_genmat)
            list_grad_comp.append(grad_comp)
        toc = time.perf_counter()
        print(f'Done serial in grad_ref_UCSF_ovlp, used time: {toc - tic}')

    funval = list_grad_comp[0]
    grad = np.array(list_grad_comp[1:])

    return funval,grad

def eval_ref_UCSF_ovlp(x,list_genmat,refvec,inivec,l_use_decomp_genmat=False):
    """
    Evaluate <ref|U|ini> as a function of the rotational angles in U
    When l_use_decomp_genmat = True, list_genmat contains decomposed genmat that gives analytical Umat
    """

   #tic = time.perf_counter()
    if l_use_decomp_genmat:
       #print(f'Do make_Umat_decomp_genmat')
        Umat = make_Umat_decomp_genmat(list_genmat,x)
    else:
        Umat, Uvec = make_and_apply_U_matrix(list_genmat,x)
   #toc = time.perf_counter()
   #print(f'Time to prepare Umat in eval_ref_UCSF_ovlp: {toc -tic}')

    func = refvec.transpose()@Umat@inivec

    func = -abs(func) #The overlap is always taken to be negative so that minimization of func gives maximization of overlap

    return func

def opt_U_for_GS_of_Hmat_v2(list_list_ex_space,Hmat_full_space,list_list_genmat,l_use_decomp_genmat,list_list_theta_ini,list_UCSF_subspace_start_end,nparal,ldisp=False,debug=False):
    """
    Optimize U parameters to get lowest ground state energy
    """

    if debug: print('\nIn opt_U_for_GS_of_Hmat_v2')

    if debug:
        E_GS_full, psi_GS_full = get_ground_state(csr_matrix(Hmat_full_space))
        print(f'E0 of full ex space: {E_GS_full}')

    n_all_theta = 0
    x0 = []
    for vec_theta_ini in list_list_theta_ini:
        x0 += list(vec_theta_ini)
       #x0 += list_theta_ini

    x0 = np.array(x0)
    n_all_theta = len(x0)
    if debug: print(f'# of all thetas: {n_all_theta}')

    tic = time.perf_counter()
    E_x0 = EGS_func_of_U_v2(Hmat_full_space,list_list_genmat,l_use_decomp_genmat,x0,list_UCSF_subspace_start_end,nparal=1,debug=True)
    toc = time.perf_counter()
    if debug: print(f'Time to evaluate EGS_func_of_U_v2: {toc - tic}')

    def cost(x):
        return EGS_func_of_U_v2(Hmat_full_space,list_list_genmat,l_use_decomp_genmat,x,list_UCSF_subspace_start_end,nparal=1,debug=False)

    options = {
        'maxiter' : 10000,
        'disp'    : ldisp,
        'gtol'    : 1.0e-4
    }

    arguments = (Hmat_full_space,list_list_genmat,l_use_decomp_genmat,list_UCSF_subspace_start_end,nparal)
    sol = minimize(gradEGS_func_of_U_v2,x0, args=arguments, method='BFGS',options=options,jac=True)

   #sol = minimize(cost, x0, method='BFGS',options=options)

    improve = sol.fun - E_x0
    print(f'Improvement over optimizing thetas: {improve}, with nparal: {nparal}')

   #Get the optimal UCSFs
    x_opt = sol.x

    ncount = 0
    nUCSF = len(list_list_genmat)
    list_UCSF = []
    list_vec_theta_opt = []
    for iUCSF in range(nUCSF):
        list_genmat = list_list_genmat[iUCSF]
        list_theta = x_opt[ncount:ncount+len(list_genmat)]
        list_pair_ex_space = list_list_ex_space[iUCSF]
       #print(len(list_genmat),len(list_list_genmat[iUCSF]),len(list_theta))
        assert len(list_genmat) == len(list_theta)
        [istart,iend] = list_UCSF_subspace_start_end[iUCSF]

        inivec = np.zeros(iend-istart)
        inivec[0] = 1.0
        if l_use_decomp_genmat:
            if nparal != 1:
                Umat = make_Umat_decomp_genmat_joblib(list_genmat,list_theta,nparal)
            else:
                Umat = make_Umat_decomp_genmat(list_genmat,list_theta)
            Uvec = Umat@inivec
        else:
            Umat, Uvec = make_and_apply_U_matrix(list_genmat,list_theta,True,inivec,False)

        ncount += len(list_genmat)
        UCSF = make_UCSF_state(list_pair_ex_space,Uvec)
        list_UCSF.append(UCSF)
        list_vec_theta_opt.append(list_theta)

    return list_vec_theta_opt, list_UCSF

def opt_U_for_GS_of_Hmat(list_list_ex_space,list_list_genmat,list_group_mp2_ampld,Enuc,obt,tbt,list_n_mp2_ampld_opt,debug=False):
    """
    Optimize U parameters to get lowest ground state energy
    """

    if debug: print('\nIn opt_U_for_GS_of_Hmat')

    nUCSF = len(list_list_ex_space)
    list_start_end = []
    if debug: print(f'# of states {nUCSF}')
    istart = 0
    x0 = []
    x_frozen = []
   #ndim_x = nUCSF*n_mp2_ampld_opt
   #ndim_x = nUCSF*len(list_group_mp2_ampld[0][0:n_mp2_ampld_opt])
   #print(f'\nlist_list_ex_space:')
   #print(list_list_ex_space)
    for iUCSF in range(nUCSF):
        ndim = len(list_list_ex_space[iUCSF])
        if debug: print(f'# of states in UCSF {iUCSF}: {ndim}')
        list_start_end.append([istart,istart+ndim])
        istart += ndim
       #print(list_group_mp2_ampld[iUCSF],n_mp2_ampld_opt)
        x0 += list_group_mp2_ampld[iUCSF][0:list_n_mp2_ampld_opt[iUCSF]]
        x_frozen += list_group_mp2_ampld[iUCSF][list_n_mp2_ampld_opt[iUCSF]:]

    if debug:
        print(f'\nlist_n_mp2_ampld_opt: {list_n_mp2_ampld_opt}')
        print(f'\nx0: {x0}')
    if debug: print(f'list_start_end: {list_start_end}')
    assert len(x_frozen) == 0 #The present setting disable freezing amplitudes.
    x0 = np.array(x0)
   #assert len(x0) == ndim_x
   #if debug: print(f'Total # of U parameters to optimize: {ndim_x}')

    ndim_all_ex = istart
    if debug: print(f'Dimension of the super Hamiltonian matrix: {ndim_all_ex}')
    if debug: print(f'\nMaking Hamiltonian super matrix for dimension {ndim_all_ex}')
    Hmat_all_ex = np.zeros([ndim_all_ex,ndim_all_ex])
   #Construct the hamiltonian matrix of the primitive basis states in list_list_ex_space
   #print(list_list_ex_space)
    for iUCSF in range(nUCSF):
        istart = list_start_end[iUCSF][0]
        for jUCSF in range(iUCSF,nUCSF):
            jstart = list_start_end[jUCSF][0]
            for ii, bas_i in enumerate(list_list_ex_space[iUCSF]):
                onlist_i = bas_i[0]
                coefs_i  = bas_i[2]
                i_ex_bas = istart + ii
               #if i_ex_bas % 10 == 0 and debug: print(f'i_ex_bas = {i_ex_bas}')
                for jj, bas_j in enumerate(list_list_ex_space[jUCSF]):
                    onlist_j = bas_j[0]
                    coefs_j  = bas_j[2]
                    j_ex_bas = jstart + jj
                    if j_ex_bas < i_ex_bas: continue
                    Helm = Helm_between_LCSDs(Enuc,obt,tbt,onlist_i,coefs_i,onlist_j,coefs_j)
                    Hmat_all_ex[i_ex_bas,j_ex_bas] = Helm
                    Hmat_all_ex[j_ex_bas,i_ex_bas] = Helm

   #EGS_func_of_U(list_list_ex_space,list_list_genmat,x0,Hmat_all_ex,list_start_end,False)
    def cost(x):
        return EGS_func_of_U(list_list_ex_space,list_list_genmat,x,x_frozen,Hmat_all_ex,list_start_end,list_n_mp2_ampld_opt,False)
    def cost_true(x):
        return EGS_func_of_U(list_list_ex_space,list_list_genmat,x,x_frozen,Hmat_all_ex,list_start_end,list_n_mp2_ampld_opt,True)
    E_GS_before_opt = cost(x0)
    if debug: print(f'E_GS before optimization: {E_GS_before_opt}')

    options = {
        'maxiter' : 10000,
        'disp'    : False,
        'gtol'    : 1.0e-4
    }

    sol = minimize(cost, x0, method='BFGS',options=options)

    nx_each_chunk = len(list_group_mp2_ampld[0])
    if debug: print(f'Optimized E_GS: {sol.fun}')

    list_Ustate_opt = []
    list_mp2_ampld_opt = []
    istart = 0
    for iUCSF in range(nUCSF):
        list_genmat = list_list_genmat[iUCSF]
        iend = istart + list_n_mp2_ampld_opt[iUCSF]
        theta_for_iUCSF = sol.x[istart:iend]
        Umat, Uvec = make_and_apply_U_matrix(list_genmat,theta_for_iUCSF)
        Ustate_opt = make_UCSF_state(list_list_ex_space[iUCSF],Uvec)
        list_Ustate_opt.append(Ustate_opt)
        list_mp2_ampld_opt.append(theta_for_iUCSF.tolist())
        istart = iend

    improve = sol.fun - E_GS_before_opt

    return list_Ustate_opt, list_mp2_ampld_opt, improve, sol.fun

def opt_multiple_U_for_GS_of_Hmat(list_list_ex_space,list_list_genmat,list_group_mp2_ampld,Enuc,obt,tbt,n_mp2_ampld_opt,list_n_U=None,debug=False):
    """
    Optimize U parameters to get lowest ground state energy
    """

    if debug: print('\nIn opt_U_for_GS_of_Hmat')


    nUCSF = len(list_list_ex_space)
    n_mp2_ampld_total = len(list_group_mp2_ampld[0][0])
    if list_n_U == None: list_n_U = [1] * nUCSF
    list_start_end = []
    if debug: print(f'# of states {nUCSF}')
    istart = 0
    x0_multi = []
    x_frozen_multi = []
    ndim_x = nUCSF*n_mp2_ampld_opt
   #ndim_x = nUCSF*len(list_group_mp2_ampld[0][0:n_mp2_ampld_opt])
    for iUCSF in range(nUCSF):
       #print(iUCSF,list_group_mp2_ampld[iUCSF],list_n_U[iUCSF])
        assert len(list_group_mp2_ampld[iUCSF]) == list_n_U[iUCSF]
        ndim = len(list_list_ex_space[iUCSF])
        if debug: print(f'# of states in UCSF {iUCSF}: {ndim}')
        list_start_end.append([istart,istart+ndim])
        istart += ndim
       #print(list_group_mp2_ampld[iUCSF])
        for group_mp2_ampld in list_group_mp2_ampld[iUCSF]:
            x0_multi += group_mp2_ampld[0:n_mp2_ampld_opt]
            x_frozen_multi += group_mp2_ampld[n_mp2_ampld_opt:]


    if debug:
        print(f'length: {len(x0_multi),len(x_frozen_multi)}')
        print(f'list_start_end: {list_start_end}')

    x0_multi = np.array(x0_multi)
    print(f'Total # of multi U parameters to optimize: {len(x0_multi)}')

    ndim_all_ex = istart
    if debug: print(f'Dimension of the super Hamiltonian matrix: {ndim_all_ex}')
    if debug: print(f'\nMaking Hamiltonian super matrix for dimension {ndim_all_ex}')
    Hmat_all_ex = np.zeros([ndim_all_ex,ndim_all_ex])
   #Construct the hamiltonian matrix of the primitive basis states in list_list_ex_space
    for iUCSF in range(nUCSF):
        istart = list_start_end[iUCSF][0]
        for jUCSF in range(iUCSF,nUCSF):
            jstart = list_start_end[jUCSF][0]
            for ii, bas_i in enumerate(list_list_ex_space[iUCSF]):
                onlist_i = bas_i[0]
                coefs_i  = bas_i[2]
                i_ex_bas = istart + ii
               #if i_ex_bas % 10 == 0 and debug: print(f'i_ex_bas = {i_ex_bas}')
                for jj, bas_j in enumerate(list_list_ex_space[jUCSF]):
                    onlist_j = bas_j[0]
                    coefs_j  = bas_j[2]
                    j_ex_bas = jstart + jj
                    if j_ex_bas < i_ex_bas: continue
                    Helm = Helm_between_LCSDs(Enuc,obt,tbt,onlist_i,coefs_i,onlist_j,coefs_j)
                    Hmat_all_ex[i_ex_bas,j_ex_bas] = Helm
                    Hmat_all_ex[j_ex_bas,i_ex_bas] = Helm

   #EGS_func = EGS_func_of_U_multiset(list_list_ex_space,list_list_genmat,x0_multi,x_frozen_multi,list_n_U,Hmat_all_ex,list_start_end,False)
   #print(f'EGS_func = {EGS_func}')
    def cost(x):
        return EGS_func_of_U_multiset(list_list_ex_space,list_list_genmat,x,x_frozen_multi,list_n_U,Hmat_all_ex,list_start_end,False)
    print(f'E_GS before optimization: {cost(x0_multi)}')

    options = {
        'maxiter' : 10000,
        'disp'    : True,
        'gtol'    : 1.0e-4
    }

    sol = minimize(cost, x0_multi, method='BFGS',options=options)

   #nx_each_chunk = len(list_group_mp2_ampld[0][0])
    nchunk = np.sum(np.array(list_n_U))
    nx_each_chunk = len(x0_multi) // nchunk
    nx_frozen_each_chunk = len(x_frozen_multi) // nchunk

    list_Ustate_opt = []
    list_mp2_ampld_opt = []
    ichunk = -1
    for iUCSF in range(nUCSF):
        list_genmat = list_list_genmat[iUCSF]
        list_opt_theta = []
        for iset in range(list_n_U[iUCSF]):
            ichunk += 1
            theta_list = sol.x[ichunk*nx_each_chunk:(ichunk+1)*nx_each_chunk].tolist() +\
                         x_frozen_multi[ichunk*nx_frozen_each_chunk:(ichunk+1)*nx_frozen_each_chunk]
            list_opt_theta.append(theta_list)
            if debug:
                print(f'iUCSF: {iUCSF}, iset: {iset}')
                print(f'theta_list: {theta_list}')

            if iset == 0:
                Umat, Uvec = make_and_apply_U_matrix(list_genmat,theta_list)
            else:
                Umat, Uvec = make_and_apply_U_matrix(list_genmat,theta_list,True,Uvec)

        Ustate_opt = make_UCSF_state(list_list_ex_space[iUCSF],Uvec)
        list_Ustate_opt.append(Ustate_opt)
        list_mp2_ampld_opt.append(list_opt_theta)

        if debug: print(f'list_mp2_ampld_opt: {list_mp2_ampld_opt}')
    return list_Ustate_opt, list_mp2_ampld_opt, sol.fun

def EGS_func_of_U_v2(Hmat_all_ex,list_list_genmat,l_use_decomp_genmat,x,list_start_end,nparal=1,debug=False):

    if debug: print(f'In EGS_func_of_U_v2')

    nUCSF = len(list_list_genmat)
    ndim_all_ex = Hmat_all_ex.shape[0]
    O_rowEX_colUCSF = np.zeros([ndim_all_ex,nUCSF])
    ncount = 0
    for iUCSF in range(nUCSF):
        list_genmat = list_list_genmat[iUCSF]
        [istart,iend] = list_start_end[iUCSF]
        list_theta = x[ncount:ncount+len(list_genmat)]
       #print(len(list_genmat),len(list_list_genmat[iUCSF]),len(list_theta))
        assert len(list_genmat) == len(list_theta)
        if debug: print(f'UCSF {iUCSF},istart {istart}, iend {iend}')

        inivec = np.zeros(iend-istart)
        inivec[0] = 1.0
        if l_use_decomp_genmat:
            if len(inivec) == 1:
                Uvec = inivec
            else:
                if nparal != 1:
                    Umat = make_Umat_decomp_genmat_joblib(list_genmat,list_theta,nparal)
                else:
                    Umat = make_Umat_decomp_genmat(list_genmat,list_theta)
                Uvec = Umat@inivec
        else:
            Umat, Uvec = make_and_apply_U_matrix(list_genmat,list_theta,True,inivec,False)

        ncount += len(list_genmat)
        O_rowEX_colUCSF[istart:iend,iUCSF] = Uvec

    Hmat_UCSF = O_rowEX_colUCSF.transpose()@Hmat_all_ex@O_rowEX_colUCSF
    if debug:
        print(f'\nHmat_UCSF in EGS_func_of_U')
        print_matrix(Hmat_UCSF)

    E_GS, psi_GS = get_ground_state(csr_matrix(Hmat_UCSF))
    if debug: print(f'E_GS of Hmat_UCSF: {E_GS}')


    return E_GS

def gradEGS_func_of_U_v2(x,Hmat_all_ex,list_list_genmat,l_use_decomp_genmat,list_start_end,nparal):

    n_param = len(x)
    list_grad_comp = []
    l_paral = False
    if nparal > 1: l_paral = True
    if l_paral:
       #Good parallel. Comment out the lines between Good parallel and End of good parallel and comment on the subsequent
       #line of list_grad_comp = Parallel(n_jobs=nparal) to return to the original parallel code
        n_comp = len(x)+1
        chunks = even_distrib_ncomp_to_nparal(n_comp,nparal)
        nchunk = len(chunks)
       #nscale = 1
       #chunk_size = 0
       #while chunk_size < 1:
       #    nchunk = nparal // nscale
       #    if nchunk < 1: nchunk = 1
       #    chunk_size = n_comp // nchunk
       #    nscale *= 2
       #list_all_idx = []
       #for ii in range(n_comp): list_all_idx.append(ii)
       #chunks = []
       #for ii in range(0,nchunk):
       #    chunks.append(list_all_idx[ii*chunk_size:(ii+1)*chunk_size])
       #chunks[-1] += list_all_idx[nchunk*chunk_size:]
        list_Hmat_all_ex = []
        list_list_list_genmat = []
        list_l_use_decomp_genmat = []
        list_x = []
        list_list_start_end = []
        for ii in range(nchunk):
            list_Hmat_all_ex.append(Hmat_all_ex)
            list_list_list_genmat.append(list_list_genmat)
            list_l_use_decomp_genmat.append(l_use_decomp_genmat)
            list_x.append(x)
            list_list_start_end.append(list_start_end)
        def grad_chunk_comp_EGS_func_of_U_v2(list_idx,Hmat_all_ex,list_list_genmat,l_use_decomp_genmat,x,list_start_end):
            list_grad_comp_1chunk = []
            for icomp in list_idx:
                grad_comp = grad_comp_EGS_func_of_U_v2(icomp,Hmat_all_ex,list_list_genmat,l_use_decomp_genmat,x,list_start_end)
                list_grad_comp_1chunk.append(grad_comp)

            return list_grad_comp_1chunk

       #list_list_grad_comp = Parallel(n_jobs=nchunk)(delayed(grad_chunk_comp_EGS_func_of_U_v2)(chunk,Hmat_all_ex,list_list_genmat,l_use_decomp_genmat,x,list_start_end) for chunk in chunks)
        list_list_grad_comp = Parallel(n_jobs=nchunk)(delayed(grad_chunk_comp_EGS_func_of_U_v2)(chunks[ii],list_Hmat_all_ex[ii],list_list_list_genmat[ii],list_l_use_decomp_genmat[ii],list_x[ii],list_list_start_end[ii]) for ii in range(nchunk))
        list_grad_comp = []
        for item in list_list_grad_comp: list_grad_comp += item
       #End of good parallel
       #list_grad_comp = Parallel(n_jobs=nparal)(delayed(grad_comp_EGS_func_of_U_v2)(ii,Hmat_all_ex,list_list_genmat,l_use_decomp_genmat,x,list_start_end) for ii in range(n_param+1))
    else:
        for icomp in range(n_param+1): #icomp = 0 returns the function value, icomp = 1 to n_param returns the gradient components
            component = grad_comp_EGS_func_of_U_v2(icomp,Hmat_all_ex,list_list_genmat,l_use_decomp_genmat,x,list_start_end)
            list_grad_comp.append(component)

    funval = list_grad_comp[0]
    fungrd = np.array(list_grad_comp[1:])

    return funval, fungrd

def grad_comp_EGS_func_of_U_v2(icomp,Hmat_all_ex,list_list_genmat,l_use_decomp_genmat,x,list_start_end):

    eps = 1e-5
    if icomp == 0:
        return EGS_func_of_U_v2(Hmat_all_ex,list_list_genmat,l_use_decomp_genmat,x,list_start_end)
    else:
        x_disp = copy.deepcopy(x)
        x_disp[icomp-1] = x[icomp-1] + eps
        EGS_plus  = EGS_func_of_U_v2(Hmat_all_ex,list_list_genmat,l_use_decomp_genmat,x_disp,list_start_end)
        x_disp[icomp-1] = x[icomp-1] - eps
        EGS_minus = EGS_func_of_U_v2(Hmat_all_ex,list_list_genmat,l_use_decomp_genmat,x_disp,list_start_end)
        grad_comp = (EGS_plus - EGS_minus) / (2.0 * eps)
        return grad_comp

def EGS_func_of_U(list_list_ex_space,list_list_genmat,x,x_frozen,Hmat_all_ex,list_start_end,list_n_mp2_ampld_opt,debug=False):

    if debug: print('\nIn EGS_func_of_U')
    nUCSF = len(list_list_ex_space)
    ndim_all_ex = Hmat_all_ex.shape[0]
   #nx_each_chunk = len(x) // nUCSF
   #nx_fronzen_each_chunk = len(x_frozen) // nUCSF
    O_rowEX_colUCSF = np.zeros([ndim_all_ex,nUCSF])
   #if debug: print(nx_each_chunk)
    istart = 0
    for iUCSF in range(nUCSF):
        list_genmat = list_list_genmat[iUCSF]
        iend = istart + list_n_mp2_ampld_opt[iUCSF]
       #x_all = x[iUCSF*nx_each_chunk:(iUCSF+1)*nx_each_chunk].tolist() \
       #      + x_frozen[iUCSF*nx_fronzen_each_chunk:(iUCSF+1)*nx_fronzen_each_chunk]
        x_all = x[istart:iend]
        if debug: print(f'UCSF {iUCSF}, x_all: {x_all}')
       #Umat, Uvec = make_and_apply_U_matrix(list_genmat,x[iUCSF*nx_each_chunk:(iUCSF+1)*nx_each_chunk])
       #if debug and iUCSF == 1:
       #    print(f'genmats of iUCSF = 1 in EGS_func_of_U')
       #    for ii, item in enumerate(list_genmat):
       #        print(f'genmat {ii}')
       #        print(item)
        Umat, Uvec = make_and_apply_U_matrix(list_genmat,x_all)
       #if debug:
       #    print(f'Uvec for UCSF{iUCSF}: {Uvec}')
       #    if iUCSF == 1:
       #        print(f'Umat of iUCSF 1:')
       #        print_matrix(Umat)
        [chunk_start, chunk_end] = list_start_end[iUCSF]
        O_rowEX_colUCSF[chunk_start:chunk_end,iUCSF] = Uvec
        istart = iend

    Hmat_UCSF = O_rowEX_colUCSF.transpose()@Hmat_all_ex@O_rowEX_colUCSF
    if debug:
        print(f'\nHmat_UCSF in EGS_func_of_U')
        print_matrix(Hmat_UCSF)
    E_GS, psi_GS = get_ground_state(Hmat_UCSF)
    if debug: print(f'E_GS of Hmat_UCSF: {E_GS}')


    return E_GS

def EGS_func_of_U_multiset(list_list_ex_space,list_list_genmat,x,x_frozen,list_n_U,Hmat_all_ex,list_start_end,debug=False):

    if debug: print('\nIn EGS_func_of_U')
    nUCSF = len(list_list_ex_space)
    ndim_all_ex = Hmat_all_ex.shape[0]
    n_chunk = np.sum(np.array(list_n_U))
    nx_each_chunk = len(x) // n_chunk
    nx_fronzen_each_chunk = len(x_frozen) // n_chunk
   #print(n_chunk,nx_each_chunk,nx_fronzen_each_chunk)
    O_rowEX_colUCSF = np.zeros([ndim_all_ex,nUCSF])
    if debug: print(nx_each_chunk)
    ichunk = -1
    for iUCSF in range(nUCSF):
        list_genmat = list_list_genmat[iUCSF]
        for iset in range(list_n_U[iUCSF]):
            ichunk += 1
            x_all = x[ichunk*nx_each_chunk:(ichunk+1)*nx_each_chunk].tolist() \
              + x_frozen[ichunk*nx_fronzen_each_chunk:(ichunk+1)*nx_fronzen_each_chunk]
       #Umat, Uvec = make_and_apply_U_matrix(list_genmat,x[iUCSF*nx_each_chunk:(iUCSF+1)*nx_each_chunk])
            if iset == 0:
                Umat, Uvec = make_and_apply_U_matrix(list_genmat,x_all)
            else:
                Umat, Uvec = make_and_apply_U_matrix(list_genmat,x_all,True,Uvec)

        [chunk_start, chunk_end] = list_start_end[iUCSF]
        O_rowEX_colUCSF[chunk_start:chunk_end,iUCSF] = Uvec

    Hmat_UCSF = O_rowEX_colUCSF.transpose()@Hmat_all_ex@O_rowEX_colUCSF
    E_GS, psi_GS = get_ground_state(Hmat_UCSF)
    if debug: print(f'E_GS of Hmat_UCSF: {E_GS}')


    return E_GS

def opt_orbitals_for_GS_of_Hmat(list_list_ex_space,list_list_genmat,list_group_mp2_ampld,Enuc,nelec,obt_spatial,tbt_phys_spatial,list_orb_rot,x0=[],debug=False):
    """
    Optimize U parameters to get lowest ground state energy
    """

    if debug: print('\nIn opt_orbitals_for_GS_of_Hmat')


    nUCSF = len(list_list_ex_space)
    n_mp2_ampld_total = len(list_group_mp2_ampld[0][0])
    list_start_end = []
    if debug: print(f'# of states {nUCSF}')
    istart = 0
    list_n_U = []
    for iUCSF in range(nUCSF):
        list_n_U.append(len(list_group_mp2_ampld[iUCSF]))
        if debug: print(iUCSF,list_group_mp2_ampld[iUCSF],list_n_U[iUCSF])
        ndim = len(list_list_ex_space[iUCSF])
        if debug: print(f'# of states in UCSF {iUCSF}: {ndim}')
        list_start_end.append([istart,istart+ndim])
        istart += ndim


    if debug:
        print(f'list_start_end: {list_start_end}')

    if len(x0) == 0: x0 = np.zeros(nparam)
    obt, tbt = orthogonal_transform_obt_tbt(x0,list_orb_rot,obt_spatial,tbt_phys_spatial)


    ndim_all_ex = istart
    if debug: print(f'Dimension of the super Hamiltonian matrix: {ndim_all_ex}')
    if debug: print(f'\nMaking Hamiltonian super matrix for dimension {ndim_all_ex}')
    Hmat_all_ex = np.zeros([ndim_all_ex,ndim_all_ex])
    O_rowEX_colUCSF = np.zeros([ndim_all_ex,nUCSF])
   #Construct the hamiltonian matrix of the primitive basis states in list_list_ex_space
   #Construct the reduced density matrices for each pair of bra and ket
    list_list_rdm1 = []
    list_list_rdm2 = []
    ilist_rdm = -1
    for iUCSF in range(nUCSF):
       #Also construct U rotational vectors
        list_genmat = list_list_genmat[iUCSF]
        for iset in range(list_n_U[iUCSF]):
            group_mp2_ampld = list_group_mp2_ampld[iUCSF][iset]
            if iset == 0:
                Umat, Uvec = make_and_apply_U_matrix(list_genmat,group_mp2_ampld)
            else:
                Umat, Uvec = make_and_apply_U_matrix(list_genmat,group_mp2_ampld,True,Uvec)

        [chunk_start, chunk_end] = list_start_end[iUCSF]
        O_rowEX_colUCSF[chunk_start:chunk_end,iUCSF] = Uvec

        istart = list_start_end[iUCSF][0]
        for jUCSF in range(iUCSF,nUCSF):
            jstart = list_start_end[jUCSF][0]
            for ii, bas_i in enumerate(list_list_ex_space[iUCSF]):
                onlist_i = bas_i[0]
                coefs_i  = bas_i[2]
                i_ex_bas = istart + ii
               #if i_ex_bas % 10 == 0 and debug: print(f'i_ex_bas = {i_ex_bas}')
                for jj, bas_j in enumerate(list_list_ex_space[jUCSF]):
                    onlist_j = bas_j[0]
                    coefs_j  = bas_j[2]
                    j_ex_bas = jstart + jj
                    if j_ex_bas < i_ex_bas: continue
                    Helm, list_rdm1, list_rdm2 = Helm_between_LCSDs(Enuc,obt,tbt,onlist_i,coefs_i,onlist_j,coefs_j,lrdm=True)
                    Hmat_all_ex[i_ex_bas,j_ex_bas] = Helm
                    Hmat_all_ex[j_ex_bas,i_ex_bas] = Helm
                    Helm_from_rdm = elm_from_list_rdm(list_rdm1,list_rdm2,Enuc/nelec,obt,tbt)
                    assert np.isclose(Helm,Helm_from_rdm)
                    ilist_rdm += 1
                   #print(f'In storing list_list_rdm: {iUCSF,istart,ii,jUCSF,jstart,jj,i_ex_bas,j_ex_bas,ilist_rdm}')
                    list_list_rdm1.append(list_rdm1)
                    list_list_rdm2.append(list_rdm2)

    if debug:
        Hmat_UCSF = O_rowEX_colUCSF.transpose()@Hmat_all_ex@O_rowEX_colUCSF
        print(f'Hmat of UCSF basis:')
        print_matrix(Hmat_UCSF)

    if debug:
        ilist_rdm = -1
        for iUCSF in range(nUCSF):
            istart = list_start_end[iUCSF][0]
            for jUCSF in range(iUCSF,nUCSF):
                jstart = list_start_end[jUCSF][0]
                for ii, bas_i in enumerate(list_list_ex_space[iUCSF]):
                    i_ex_bas = istart + ii
                    for jj, bas_j in enumerate(list_list_ex_space[jUCSF]):
                        j_ex_bas = jstart + jj
                        if j_ex_bas < i_ex_bas: continue
                        ilist_rdm += 1
                        list_rdm1 = list_list_rdm1[ilist_rdm]
                        list_rdm2 = list_list_rdm2[ilist_rdm]
                        Helm_from_rdm = elm_from_list_rdm(list_rdm1,list_rdm2,Enuc/nelec,obt,tbt)
                        assert np.isclose(Helm_from_rdm,Hmat_all_ex[i_ex_bas,j_ex_bas])


    nparam = len(list_orb_rot)

    if debug: print(f'\n# of orbital rotation angles to optimize: {nparam}')


    def cost(x):
        return EGS_func_of_orbrot(list_list_rdm1,list_list_rdm2,ndim_all_ex,x,list_orb_rot,Enuc,nelec,obt_spatial,tbt_phys_spatial,O_rowEX_colUCSF,list_start_end,list_list_ex_space)
    tic = time.perf_counter()
    func_ini = costval = cost(x0)
    toc = time.perf_counter()
    print(f'Time to one function evaluation: {toc - tic}')
    print(f'test cost function: {func_ini}')

    options = {
        'maxiter' : 10000,
        'disp'    : True,
        'gtol'    : 1.0e-4
    }

    sol = minimize(cost, x0, method='BFGS',options=options)

    if debug: print(f'opt rotational angles: {sol.x}')
    print(f'Energy lowering due to orbital rotaiton: {sol.fun - func_ini}')

    obt_opt_spin, tbt_opt_spin = orthogonal_transform_obt_tbt(sol.x,list_orb_rot,obt_spatial,tbt_phys_spatial)

   #Create the spatial orbital integrals of the transformed orbitals
   #n_spatialmo = obt_phys_spatial.shape[0]

   #kappa_mat = np.zeros([n_spatialmo,n_spatialmo])
   #for ix, pair in enumerate(list_orb_rot):
   #    [iorb,jorb] = pair
   #    kappa_mat[iorb,jorb] = sol.x[ix]
   #    kappa_mat[jorb,iorb] = -sol.x[ix]

   #Omat = scipy.linalg.expm(kappa_mat)

   #obt_phys_spatial_trans = np.einsum('pq,pa,qb->ab',obt_phys_spatial,Omat,Omat,optimize=True)
   #tbt_phys_spatial_trans = np.einsum('pqrs,pa,qb,rc,sd->abcd',tbt_phys_spatial,Omat,Omat,Omat,Omat,optimize=True)

   #obt_phys_spin_trans = obt_phys_spatial_to_spin(obt_phys_spatial_trans)
   #tbt_phys_spin_trans = tbt_phys_spatial_to_spin(tbt_phys_spatial_trans)

    return sol.x, sol.fun, obt_opt_spin, tbt_opt_spin

def opt_orbitals_for_GS_of_Hmat_1setU_each_UCSF(list_list_ex_space,list_list_genmat,list_group_mp2_ampld,Enuc,nelec,obt_spatial,tbt_phys_spatial,list_orb_rot,x0=[],debug=False):
    """
    Optimize U parameters to get lowest ground state energy
    """

    if debug: print('\nIn opt_orbitals_for_GS_of_Hmat')


    nUCSF = len(list_list_ex_space)
    list_start_end = []
    if debug: print(f'# of states {nUCSF}')
    istart = 0
    for iUCSF in range(nUCSF):
        ndim = len(list_list_ex_space[iUCSF])
        if debug: print(f'# of states in UCSF {iUCSF}: {ndim}')
        list_start_end.append([istart,istart+ndim])
        istart += ndim


    if debug:
        print(f'list_start_end: {list_start_end}')

    if len(x0) == 0: x0 = np.zeros(nparam)
    obt, tbt = orthogonal_transform_obt_tbt(x0,list_orb_rot,obt_spatial,tbt_phys_spatial)


    ndim_all_ex = istart
    if debug: print(f'Dimension of the super Hamiltonian matrix: {ndim_all_ex}')
    if debug: print(f'\nMaking Hamiltonian super matrix for dimension {ndim_all_ex}')
    Hmat_all_ex = np.zeros([ndim_all_ex,ndim_all_ex])
    O_rowEX_colUCSF = np.zeros([ndim_all_ex,nUCSF])
   #Construct the hamiltonian matrix of the primitive basis states in list_list_ex_space
   #Construct the reduced density matrices for each pair of bra and ket
    list_list_rdm1 = []
    list_list_rdm2 = []
    ilist_rdm = -1
    for iUCSF in range(nUCSF):
       #Also construct U rotational vectors
        list_genmat = list_list_genmat[iUCSF]
        group_mp2_ampld = list_group_mp2_ampld[iUCSF]
        Umat, Uvec = make_and_apply_U_matrix(list_genmat,group_mp2_ampld)

        [chunk_start, chunk_end] = list_start_end[iUCSF]
        O_rowEX_colUCSF[chunk_start:chunk_end,iUCSF] = Uvec

        istart = list_start_end[iUCSF][0]
        for jUCSF in range(iUCSF,nUCSF):
            jstart = list_start_end[jUCSF][0]
            for ii, bas_i in enumerate(list_list_ex_space[iUCSF]):
                onlist_i = bas_i[0]
                coefs_i  = bas_i[2]
                i_ex_bas = istart + ii
               #if i_ex_bas % 10 == 0 and debug: print(f'i_ex_bas = {i_ex_bas}')
                for jj, bas_j in enumerate(list_list_ex_space[jUCSF]):
                    onlist_j = bas_j[0]
                    coefs_j  = bas_j[2]
                    j_ex_bas = jstart + jj
                    if j_ex_bas < i_ex_bas: continue
                    Helm, list_rdm1, list_rdm2 = Helm_between_LCSDs(Enuc,obt,tbt,onlist_i,coefs_i,onlist_j,coefs_j,lrdm=True)
                    Hmat_all_ex[i_ex_bas,j_ex_bas] = Helm
                    Hmat_all_ex[j_ex_bas,i_ex_bas] = Helm
                    Helm_from_rdm = elm_from_list_rdm(list_rdm1,list_rdm2,Enuc/nelec,obt,tbt)
                    assert np.isclose(Helm,Helm_from_rdm)
                    ilist_rdm += 1
                   #print(f'In storing list_list_rdm: {iUCSF,istart,ii,jUCSF,jstart,jj,i_ex_bas,j_ex_bas,ilist_rdm}')
                    list_list_rdm1.append(list_rdm1)
                    list_list_rdm2.append(list_rdm2)

    if debug:
        Hmat_UCSF = O_rowEX_colUCSF.transpose()@Hmat_all_ex@O_rowEX_colUCSF
        print(f'Hmat of UCSF basis:')
        print_matrix(Hmat_UCSF)

    if debug:
        ilist_rdm = -1
        for iUCSF in range(nUCSF):
            istart = list_start_end[iUCSF][0]
            for jUCSF in range(iUCSF,nUCSF):
                jstart = list_start_end[jUCSF][0]
                for ii, bas_i in enumerate(list_list_ex_space[iUCSF]):
                    i_ex_bas = istart + ii
                    for jj, bas_j in enumerate(list_list_ex_space[jUCSF]):
                        j_ex_bas = jstart + jj
                        if j_ex_bas < i_ex_bas: continue
                        ilist_rdm += 1
                        list_rdm1 = list_list_rdm1[ilist_rdm]
                        list_rdm2 = list_list_rdm2[ilist_rdm]
                        Helm_from_rdm = elm_from_list_rdm(list_rdm1,list_rdm2,Enuc/nelec,obt,tbt)
                        assert np.isclose(Helm_from_rdm,Hmat_all_ex[i_ex_bas,j_ex_bas])


    nparam = len(list_orb_rot)

    if debug: print(f'\n# of orbital rotation angles to optimize: {nparam}')


    def cost(x):
        return EGS_func_of_orbrot(list_list_rdm1,list_list_rdm2,ndim_all_ex,x,list_orb_rot,Enuc,nelec,obt_spatial,tbt_phys_spatial,O_rowEX_colUCSF,list_start_end,list_list_ex_space)
    tic = time.perf_counter()
    func_ini = costval = cost(x0)
    toc = time.perf_counter()
    print(f'Time to one function evaluation: {toc - tic}')
    print(f'test cost function: {func_ini}')

    options = {
        'maxiter' : 10000,
        'disp'    : True,
        'gtol'    : 1.0e-4
    }

    sol = minimize(cost, x0, method='BFGS',options=options)

    if debug: print(f'opt rotational angles: {sol.x}')
    print(f'Energy lowering due to orbital rotaiton: {sol.fun - func_ini}')

    obt_opt_spin, tbt_opt_spin = orthogonal_transform_obt_tbt(sol.x,list_orb_rot,obt_spatial,tbt_phys_spatial)

   #Create the spatial orbital integrals of the transformed orbitals
   #n_spatialmo = obt_phys_spatial.shape[0]

   #kappa_mat = np.zeros([n_spatialmo,n_spatialmo])
   #for ix, pair in enumerate(list_orb_rot):
   #    [iorb,jorb] = pair
   #    kappa_mat[iorb,jorb] = sol.x[ix]
   #    kappa_mat[jorb,iorb] = -sol.x[ix]

   #Omat = scipy.linalg.expm(kappa_mat)

   #obt_phys_spatial_trans = np.einsum('pq,pa,qb->ab',obt_phys_spatial,Omat,Omat,optimize=True)
   #tbt_phys_spatial_trans = np.einsum('pqrs,pa,qb,rc,sd->abcd',tbt_phys_spatial,Omat,Omat,Omat,Omat,optimize=True)

   #obt_phys_spin_trans = obt_phys_spatial_to_spin(obt_phys_spatial_trans)
   #tbt_phys_spin_trans = tbt_phys_spatial_to_spin(tbt_phys_spatial_trans)

    return sol.x, sol.fun, obt_opt_spin, tbt_opt_spin

def opt_orbtials_for_GS_of_CSF_space(list_CSF,psi_coefs,Enuc,obt_spatial,tbt_spatial,list_orb_rot,list_list_rdm1,list_list_rdm2,l_axial_sym,list_degmo,x0=[],nparal=1,debug=False):
    """
    Optimize orbitals through rotation of specific orbital pairs for the ground state of list_CSF
    """

    if debug: print('\nIn opt_orbtials_for_GS_of_CSF_space')

    if len(x0) == 0:
        x0 = np.array([0.0]*len(list_orb_rot))
        obt = obt_phys_spatial_to_spin(obt_spatial)
        tbt = tbt_phys_spatial_to_spin(tbt_spatial)
    else:
        assert len(list_orb_rot) == len(x0)
        obt, tbt = orthogonal_transform_obt_tbt(x0,list_orb_rot,obt_spatial,tbt_spatial)

   #Construct the Hamiltonian matrix of the whole ex space and the reduced density matrices
   #Hmat_CSF, list_list_rdm1, list_list_rdm2 = construct_Hmat_CSFs(list_CSF,Enuc,obt,tbt,lrdm=True)
    if debug:
        Hmat_CSF = construct_Hmat_CSFs(list_CSF,Enuc,obt,tbt)

    nCSF = len(list_CSF)
    nelec = len(np.where(list_CSF[0][0][0] == 1.0)[0])
    print(f'nelec: {nelec}')
    if debug:
        ilist_rdm = -1
        for iCSF in range(nCSF):
            for jCSF in range(iCSF,nCSF):
                ilist_rdm += 1
                list_rdm1 = list_list_rdm1[ilist_rdm]
                list_rdm2 = list_list_rdm2[ilist_rdm]
                Helm_from_rdm = elm_from_list_rdm(list_rdm1,list_rdm2,Enuc/nelec,obt,tbt)
                if not np.isclose(Helm_from_rdm,Hmat_CSF[iCSF,jCSF]):
                    print('rdm test failed')
                    print(Helm_from_rdm,Hmat_CSF[iCSF,jCSF])
                    print(iCSF,jCSF)
                    print(list_CSF[iCSF])
                    print(list_CSF[jCSF])
                    print('rdm1')
                    print(list_rdm1)
                    print('rdm2')
                    print(list_rdm2)
                    sys.exit()

    if debug: print('rdm test passed')

    def cost(x):
       #if debug:
       #    return EGS_func_of_orbrot_with_CI_coefs(x,list_orb_rot,list_list_rdm1,list_list_rdm2,psi_coefs,Enuc,nelec,obt_spatial,tbt_spatial,list_CSF)
       #else:
            return EGS_func_of_orbrot_with_CI_coefs(x,list_orb_rot,list_list_rdm1,list_list_rdm2,psi_coefs,Enuc,nelec,obt_spatial,tbt_spatial)
    tic = time.perf_counter()
    E_before_min = cost(x0)
    toc = time.perf_counter()
    print(f'time for function evaluation: {toc - tic}')
    if debug: print(f'Function value before minimization: {E_before_min}')

    options = {
        'maxiter' : 10000,
        'disp'    : True,
        'gtol'    : 1.0e-4
    }

   #sol = minimize(cost, x0, method='BFGS',options=options)
    arguments = (list_orb_rot,list_list_rdm1,list_list_rdm2,psi_coefs,Enuc,nelec,obt_spatial,tbt_spatial,nparal)
    sol = minimize(grad_EGS_of_orbrot,x0, args=arguments, method='BFGS',options=options,jac=True)

    x_orbrot_opt = sol.x
    if l_axial_sym:
        symmetrize_xorbrot(list_orb_rot,x_orbrot_opt,list_degmo)

    obt_opt, tbt_opt = orthogonal_transform_obt_tbt(sol.x,list_orb_rot,obt_spatial,tbt_spatial)

    print(f'\nEnergy lowering of orbital rotation in this round: {sol.fun - E_before_min}')

    return sol.x, sol.fun, obt_opt, tbt_opt

def grad_EGS_of_orbrot(x_orbrot,list_orb_rot,list_list_rdm1,list_list_rdm2,psi_coefs,Enuc,nelec,obt_spatial,tbt_spatial,nparal=1):

    n_param = len(x_orbrot)
    list_grad_comp = []
    l_paral = False
    if nparal > 1: l_paral = True
    if l_paral:
       #Good Parallel
        n_comp = n_param+1
        chunks = even_distrib_ncomp_to_nparal(n_comp,nparal)
        nchunk = len(chunks)
       #chunk_size = n_comp // nparal
       #list_all_idx = []
       #for ii in range(n_comp): list_all_idx.append(ii)
       #chunks = []
       #for ii in range(0,nparal):
       #    chunks.append(list_all_idx[ii*chunk_size:(ii+1)*chunk_size])
       #chunks[-1] += list_all_idx[nparal*chunk_size:]
        def grad_chunk_comp_EGS_of_orbrot(list_idx,x_orbrot,list_orb_rot,list_list_rdm1,list_list_rdm2,psi_coefs,Enuc,nelec,obt_spatial,tbt_spatial):
            list_grad_comp_1chunk = []
            for icomp in list_idx:
                grad_comp = grad_comp_EGS_of_orbrot(icomp,x_orbrot,list_orb_rot,list_list_rdm1,list_list_rdm2,psi_coefs,Enuc,nelec,obt_spatial,tbt_spatial)
                list_grad_comp_1chunk.append(grad_comp)

            return list_grad_comp_1chunk

        list_x_orbrot = []
        list_list_orb_rot = []
        list_list_list_rdm1 = []
        list_list_list_rdm2 = []
        list_psi_coefs = []
        list_Enuc = []
        list_nelec = []
        list_obt_spatial = []
        list_tbt_spatial = []
        for ichunk in range(nchunk):
            list_x_orbrot.append(x_orbrot)
            list_list_orb_rot.append(list_orb_rot)
            list_list_list_rdm1.append(list_list_rdm1)
            list_list_list_rdm2.append(list_list_rdm2)
            list_psi_coefs.append(psi_coefs)
            list_Enuc.append(Enuc)
            list_nelec.append(nelec)
            list_obt_spatial.append(obt_spatial)
            list_tbt_spatial.append(tbt_spatial)
       #list_list_grad_comp = Parallel(n_jobs=nparal)(delayed(grad_chunk_comp_EGS_of_orbrot)(chunks[ii],x_orbrot,list_orb_rot,list_list_rdm1,list_list_rdm2,psi_coefs,Enuc,nelec,obt_spatial,tbt_spatial) for ii in range(nchunk))
        list_list_grad_comp = Parallel(n_jobs=nparal)(delayed(grad_chunk_comp_EGS_of_orbrot)(chunks[ii],list_x_orbrot[ii],list_list_orb_rot[ii],list_list_list_rdm1[ii],list_list_list_rdm2[ii],list_psi_coefs[ii],list_Enuc[ii],list_nelec[ii],list_obt_spatial[ii],list_tbt_spatial[ii]) for ii in range(nchunk))
        list_grad_comp = []
        for item in list_list_grad_comp:
           list_grad_comp += item
       #End of good parallel
       #list_grad_comp = Parallel(n_jobs=nparal)(delayed(grad_comp_EGS_of_orbrot)(ii,x_orbrot,list_orb_rot,list_list_rdm1,list_list_rdm2,psi_coefs,Enuc,nelec,obt_spatial,tbt_spatial) for ii in range(n_param+1))
    else:
        for icomp in range(n_param+1): #icomp = 0 returns the function value, icomp = 1 to n_param returns the gradient components
            component = grad_comp_EGS_of_orbrot(icomp,x_orbrot,list_orb_rot,list_list_rdm1,list_list_rdm2,psi_coefs,Enuc,nelec,obt_spatial,tbt_spatial)
            list_grad_comp.append(component)

    funval = list_grad_comp[0]
    fungrd = np.array(list_grad_comp[1:])

    return funval, fungrd

def grad_comp_EGS_of_orbrot(icomp,x_orbrot,list_orb_rot,list_list_rdm1,list_list_rdm2,psi_coefs,Enuc,nelec,obt_spatial,tbt_spatial,eps=1.0e-4):

    if icomp == 0:
        EGS = EGS_func_of_orbrot_with_CI_coefs(x_orbrot,list_orb_rot,list_list_rdm1,list_list_rdm2,psi_coefs,Enuc,nelec,obt_spatial,tbt_spatial)
        return EGS
    else:
        x_disp = x_orbrot.copy()
        x_disp[icomp-1] = x_orbrot[icomp-1]+eps
        EGS_plus = EGS_func_of_orbrot_with_CI_coefs(x_disp,list_orb_rot,list_list_rdm1,list_list_rdm2,psi_coefs,Enuc,nelec,obt_spatial,tbt_spatial)
        x_disp[icomp-1] = x_orbrot[icomp-1]-eps
        EGS_minus = EGS_func_of_orbrot_with_CI_coefs(x_disp,list_orb_rot,list_list_rdm1,list_list_rdm2,psi_coefs,Enuc,nelec,obt_spatial,tbt_spatial)
        grad_comp = (EGS_plus - EGS_minus) / (2.0 * eps)
        return grad_comp

def EGS_func_of_orbrot_with_CI_coefs(x_orbrot,list_orb_rot,list_list_rdm1,list_list_rdm2,psi_coefs,Enuc,nelec,obt_spatial,tbt_spatial,list_CSF=[]):
    """
    Given 1rdm, 2rdm, and CSF coefficients, return the energy of a state as function of orbital rotational angles
    """

    obt_trans_spin, tbt_trans_spin = orthogonal_transform_obt_tbt(x_orbrot,list_orb_rot,obt_spatial,tbt_spatial)

    nCSF = len(psi_coefs)
    E_state = 0.0
    ilist_rdm = -1
    for iCSF in range(nCSF):
        for jCSF in range(iCSF,nCSF):
            ilist_rdm += 1
            list_rdm1 = list_list_rdm1[ilist_rdm]
            list_rdm2 = list_list_rdm2[ilist_rdm]
            Helm = elm_from_list_rdm(list_rdm1,list_rdm2,Enuc/nelec,obt_trans_spin,tbt_trans_spin)
            E_state += Helm*psi_coefs[iCSF]*psi_coefs[jCSF]
            if iCSF != jCSF:
                E_state += Helm*psi_coefs[iCSF]*psi_coefs[jCSF]

    return E_state

def EGS_func_of_orbrot(list_list_rdm1,list_list_rdm2,ndim_all_ex,x_orbrot,list_orb_rot,Enuc,nelec,obt_phys_spatial,tbt_phys_spatial,O_rowEX_colUCSF,list_start_end,list_list_ex_space):

    obt_trans_spin, tbt_trans_spin = orthogonal_transform_obt_tbt(x_orbrot,list_orb_rot,obt_phys_spatial,tbt_phys_spatial)


    Hmat_all_ex=np.zeros([ndim_all_ex,ndim_all_ex])
    ilist_rdm = -1
   #for i_ex_bas in range(ndim_all_ex):
   #        for j_ex_bas in range(i_ex_bas,ndim_all_ex):
   #            ilist_rdm += 1
   #            list_rdm1 = list_list_rdm1[ilist_rdm]
   #            list_rdm2 = list_list_rdm2[ilist_rdm]
   #            Helm_from_rdm = elm_from_list_rdm(list_rdm1,list_rdm2,Enuc/nelec,obt_trans_spin,tbt_trans_spin)
   #            Hmat_all_ex[i_ex_bas,j_ex_bas] = Helm_from_rdm
   #            Hmat_all_ex[j_ex_bas,i_ex_bas] = Helm_from_rdm

    nUCSF = O_rowEX_colUCSF.shape[1]
    for iUCSF in range(nUCSF):
        istart = list_start_end[iUCSF][0]
        for jUCSF in range(iUCSF,nUCSF):
            jstart = list_start_end[jUCSF][0]
            for ii, bas_i in enumerate(list_list_ex_space[iUCSF]):
                i_ex_bas = istart + ii
                for jj, bas_j in enumerate(list_list_ex_space[jUCSF]):
                    j_ex_bas = jstart + jj
                    if j_ex_bas < i_ex_bas: continue
                    ilist_rdm += 1
                    list_rdm1 = list_list_rdm1[ilist_rdm]
                    list_rdm2 = list_list_rdm2[ilist_rdm]
                    Helm_from_rdm = elm_from_list_rdm(list_rdm1,list_rdm2,Enuc/nelec,obt_trans_spin,tbt_trans_spin)
                    Hmat_all_ex[i_ex_bas,j_ex_bas] = Helm_from_rdm
                    Hmat_all_ex[j_ex_bas,i_ex_bas] = Helm_from_rdm


    Hmat_UCSF = O_rowEX_colUCSF.transpose()@Hmat_all_ex@O_rowEX_colUCSF
   #print(f'Hmat_UCSF in EGS_func_of_orbrot')
   #print_matrix(Hmat_UCSF)
    E_GS, psi_GS = get_ground_state(Hmat_UCSF)

   #print(f'E_GS in EGS_func_of_orbrot: {E_GS}')
    return E_GS

# Set True for one run to assert the vectorized H-rebuild reproduces the
# original per-pair loop element-for-element, then set back to False for speed.
_CHECK_VECTORIZED_H = False

# Single-entry cache for the flattened RDM arrays, keyed on the *identity* of
# list_list_rdm1. The RDMs are constant for an entire orbital-optimization
# round (built once, then only x_orbrot changes), so the flat arrays are reused
# across the hundreds of energy evaluations in that round. A new round builds a
# new list object → identity differs → cache recomputes. We hold a reference to
# the key object, so there is no id()-reuse hazard.
_flat_rdm_cache = {'key': None, 'flat': None}

def _flatten_list_list_rdm(list_list_rdm1, list_list_rdm2):
    """Flatten the ragged per-pair RDM lists into flat numpy arrays so the
    Hamiltonian can be rebuilt with vectorized (C-level) numpy ops instead of
    ~n(n+1)/2 Python calls. Cached on id(list_list_rdm1); see note above.

    list_rdm1 entries are [i, j, val]      → contract as val * obt[j, i]
    list_rdm2 entries are [i, j, k, l, val]→ contract as val * tbt[k, l, j, i]
    constant per pair = (sum of val where i==j) * (Enuc/nelec)
    """
    if _flat_rdm_cache['key'] is list_list_rdm1:
        return _flat_rdm_cache['flat']

    p1, i1, j1, v1 = [], [], [], []
    dp, dv = [], []              # diagonal (i==j) entries for the constant term
    for pid, lst in enumerate(list_list_rdm1):
        for entry in lst:
            i, j, val = entry[0], entry[1], entry[2]
            p1.append(pid); i1.append(i); j1.append(j); v1.append(val)
            if i == j:
                dp.append(pid); dv.append(val)

    p2, i2, j2, k2, l2, v2 = [], [], [], [], [], []
    for pid, lst in enumerate(list_list_rdm2):
        for entry in lst:
            p2.append(pid); i2.append(entry[0]); j2.append(entry[1])
            k2.append(entry[2]); l2.append(entry[3]); v2.append(entry[4])

    ip = np.intp
    flat = {
        'npair':  len(list_list_rdm1),
        'p1': np.asarray(p1, dtype=ip), 'i1': np.asarray(i1, dtype=ip),
        'j1': np.asarray(j1, dtype=ip), 'v1': np.asarray(v1, dtype=float),
        'dp': np.asarray(dp, dtype=ip), 'dv': np.asarray(dv, dtype=float),
        'p2': np.asarray(p2, dtype=ip), 'i2': np.asarray(i2, dtype=ip),
        'j2': np.asarray(j2, dtype=ip), 'k2': np.asarray(k2, dtype=ip),
        'l2': np.asarray(l2, dtype=ip), 'v2': np.asarray(v2, dtype=float),
    }
    _flat_rdm_cache['key'] = list_list_rdm1
    _flat_rdm_cache['flat'] = flat
    return flat

def _H_upper_from_flat_rdm(flat, obt_phys, tbt_phys, const_per_electron):
    """Vectorized rebuild of the upper-triangular H values (in np.triu_indices
    order) from the flattened RDMs and the current integrals. Replaces the
    per-pair Python double loop with a few gather + segment-sum (bincount) ops."""
    npair = flat['npair']
    Hflat = np.zeros(npair, dtype=float)
    if flat['v1'].size:                                    # 1-electron: val * obt[j, i]
        t1 = flat['v1'] * obt_phys[flat['j1'], flat['i1']]
        Hflat += np.bincount(flat['p1'], weights=t1, minlength=npair)
    if flat['v2'].size:                                    # 2-electron: val * tbt[k, l, j, i]
        t2 = flat['v2'] * tbt_phys[flat['k2'], flat['l2'], flat['j2'], flat['i2']]
        Hflat += np.bincount(flat['p2'], weights=t2, minlength=npair)
    if flat['dv'].size:                                    # constant: (diag rdm1 trace) * const
        Hflat += np.bincount(flat['dp'], weights=flat['dv'], minlength=npair) * const_per_electron
    return Hflat

def weighted_n_roots_func_of_orbrot(x_orbrot,n_roots,weights,list_orb_rot,list_list_rdm1,list_list_rdm2,Enuc,nelec,obt_spatial,
    tbt_spatial,normalize_weights=True):
    """
    Weighted objective over the lowest n_roots eigenvalues of H(x) in the given basis.

    J(x) = sum_{k=1..n_roots} w_k * E_k(x)

    If normalize_weights=True, weights are rescaled so sum(w)=1.
    """

    w = np.array(weights, dtype=float).copy()
    if w.ndim != 1:
        raise ValueError("weights must be 1D")
    if len(w) < n_roots:
        raise ValueError(f"weights length {len(w)} < n_roots {n_roots}")
    w = w[:n_roots]

    if normalize_weights:
        s = np.sum(w)
        if np.isclose(s, 0.0):
            raise ValueError("sum(weights) is 0; cannot normalize")
        w = w / s

    obt_trans_spin, tbt_trans_spin = orthogonal_transform_obt_tbt(x_orbrot, list_orb_rot, obt_spatial, tbt_spatial)

    n_pairs = len(list_list_rdm1)
    n = int((np.sqrt(8.0 * n_pairs + 1.0) - 1.0) / 2.0)
    if n * (n + 1) // 2 != n_pairs:
        raise ValueError(
            f"RDM list length {n_pairs} is not triangular. Inferred n={n}."
        )

    # Vectorized rebuild of H: the integrals (obt_trans_spin/tbt_trans_spin) are
    # shared across all n(n+1)/2 pairs; only the per-pair RDMs differ, and H[I,J]
    # is linear in the integrals. So flatten every RDM entry once (cached per
    # round) and contract in bulk instead of looping over pairs in Python.
    flat = _flatten_list_list_rdm(list_list_rdm1, list_list_rdm2)
    Hflat = _H_upper_from_flat_rdm(flat, obt_trans_spin, tbt_trans_spin, Enuc / nelec)
    H = np.zeros((n, n), dtype=float)
    iu = np.triu_indices(n)                 # same row-major order as the original ilist walk
    H[iu] = Hflat
    H = H + H.T - np.diag(np.diag(H))        # symmetric fill (diagonal counted once)

    if _CHECK_VECTORIZED_H:
        # One-off correctness gate: reproduce H with the original per-pair loop
        # and assert equality. Flip _CHECK_VECTORIZED_H off once verified.
        H_ref = np.zeros((n, n), dtype=float)
        ilist = -1
        for i in range(n):
            for j in range(i, n):
                ilist += 1
                Hij = elm_from_list_rdm(list_list_rdm1[ilist], list_list_rdm2[ilist],
                                        Enuc / nelec, obt_trans_spin, tbt_trans_spin)
                H_ref[i, j] = Hij
                H_ref[j, i] = Hij
        assert np.allclose(H, H_ref, atol=1e-10), \
            f'vectorized H mismatch: max|ΔH|={np.max(np.abs(H - H_ref)):.3e}'

    n_eff = min(n_roots, n)
    if n_eff < n_roots:
        w = w[:n_eff]
        if normalize_weights:
            w = w / np.sum(w)

    if n_eff < n:
        # Only the lowest n_eff eigenvalues are needed — use the sparse
        # solver (ARPACK, k < n) instead of a full dense diagonalization of
        # the whole n×n matrix. Big win when n >> n_eff (e.g. n=1237, n_eff=1).
        evals = get_lowest_n_eigen(csr_matrix(H), n_eff)
    else:
        # n_eff == n (tiny matrix): ARPACK can't return all eigenvalues, so
        # fall back to the dense solver.
        evals = np.linalg.eigvalsh(H)
        evals.sort()
    return float(np.dot(w, evals[:n_eff]))

def grad_comp_weighted_n_roots_of_orbrot(icomp,x_orbrot,eps,n_roots,weights,list_orb_rot,list_list_rdm1,list_list_rdm2,
                                         Enuc,nelec,obt_spatial,tbt_spatial,normalize_weights=True):
    if icomp == 0:
        return weighted_n_roots_func_of_orbrot(x_orbrot,n_roots,weights, list_orb_rot,list_list_rdm1,list_list_rdm2,Enuc,
                                               nelec,obt_spatial,tbt_spatial,normalize_weights=normalize_weights,)

    k = icomp - 1
    x_plus = x_orbrot.copy()
    x_minus = x_orbrot.copy()
    x_plus[k] += eps
    x_minus[k] -= eps

    f_plus = weighted_n_roots_func_of_orbrot(x_plus,n_roots,weights, list_orb_rot,list_list_rdm1,list_list_rdm2,Enuc,
                                               nelec,obt_spatial,tbt_spatial,normalize_weights=normalize_weights)
    f_minus = weighted_n_roots_func_of_orbrot(x_minus,n_roots,weights, list_orb_rot,list_list_rdm1,list_list_rdm2,Enuc,
                                               nelec,obt_spatial,tbt_spatial,normalize_weights=normalize_weights)
    return (f_plus - f_minus) / (2.0 * eps)

def grad_weighted_n_roots_of_orbrot(x_orbrot,n_roots,weights,list_orb_rot,list_list_rdm1,list_list_rdm2,Enuc,nelec,obt_spatial,tbt_spatial,nparal=1,
    eps=1.0e-4,normalize_weights=True):
    """
    Returns (fun, grad) for BFGS: fun is weighted sum of lowest n_roots eigenvalues,
    grad is finite-difference gradient.
    """

    n_param = len(x_orbrot)
    n_comp = n_param + 1  # fun + each grad component

    if nparal > 1:
        idx = list(range(n_comp))
        chunks = np.array_split(idx, nparal)

        def do_chunk(chunk):
            out = []
            for icomp in chunk:
                out.append(grad_comp_weighted_n_roots_of_orbrot(icomp, x_orbrot, eps, n_roots, weights,list_orb_rot, list_list_rdm1, list_list_rdm2,
                        Enuc, nelec, obt_spatial, tbt_spatial,normalize_weights=normalize_weights))
            return out

        parts = Parallel(n_jobs=nparal, prefer='threads')(delayed(do_chunk)(chunk.tolist()) for chunk in chunks)
        comps = []
        for p in parts:
            comps.extend(p)
    else:
        comps = [grad_comp_weighted_n_roots_of_orbrot(icomp, x_orbrot, eps,n_roots, weights,list_orb_rot, list_list_rdm1,
                list_list_rdm2, Enuc, nelec, obt_spatial, tbt_spatial,normalize_weights=normalize_weights) for icomp in range(n_comp)]
    fun = comps[0]
    grad = np.array(comps[1:], dtype=float)
    return fun, grad

def opt_orbitals_for_weighted_n_roots_of_CSF_space(list_CSF,n_roots,Enuc,obt_spatial,tbt_spatial,list_orb_rot,list_list_rdm1,list_list_rdm2,
    l_axial_sym,list_degmo,x0=None,nparal=1,eps=1e-4,normalize_weights=True,debug=False,linked_orbrot_pairs=None):
    """
    Optimize orbital rotation angles x_orbrot to minimize:
        J(x) = sum_{k=1..n_roots} w_k * E_k(x)
    where E_k are the lowest eigenvalues of H(x) in the provided basis (e.g. UCSF basis).
    If linked_orbrot_pairs is provided (list of (k1,k2) index pairs), x[k1] is forced equal
    to x[k2] at every gradient step to preserve D∞h degeneracy.
    """

    weights = [1.0 / n_roots] * n_roots

    if x0 is None or len(x0) == 0:
        x0 = np.zeros(len(list_orb_rot), dtype=float)
    else:
        x0 = np.array(x0, dtype=float)
        if len(x0) != len(list_orb_rot):
            raise ValueError("len(x0) must equal len(list_orb_rot)")

    # Infer nelec the same way you do (adjust if UCSF object format differs)
    try:
        nelec = len(np.where(list_CSF[0][0][0] == 1.0)[0])
    except Exception:
        raise ValueError(
            "Could not infer nelec from list_CSF[0][0][0]. "
        )

    # Pre-warm the flattened-RDM cache once, single-threaded, so the first
    # (parallel) gradient evaluation doesn't have every worker redundantly
    # flatten the RDMs at the same time.
    _flatten_list_list_rdm(list_list_rdm1, list_list_rdm2)

    def _project_linked(x):
        """Average linked pairs so they stay equal."""
        if linked_orbrot_pairs:
            x = x.copy()
            for k1, k2 in linked_orbrot_pairs:
                avg = 0.5 * (x[k1] + x[k2])
                x[k1] = avg
                x[k2] = avg
        return x

    def fun_and_grad(x):
        x = _project_linked(x)
        f, g = grad_weighted_n_roots_of_orbrot(x, n_roots, weights, list_orb_rot, list_list_rdm1, list_list_rdm2,
            Enuc, nelec, obt_spatial, tbt_spatial, nparal=nparal, eps=eps, normalize_weights=normalize_weights)
        g = _project_linked(g)  # project gradient to linked subspace
        return f, g

    tic = time.perf_counter()
    f0, g0 = fun_and_grad(x0)
    toc = time.perf_counter()
    print(f"time for objective+grad evaluation: {toc - tic}")
    if debug:
        print(f"Initial objective: {f0}")
        print(f"Initial grad norm: {np.linalg.norm(g0)}")

    sol = minimize(lambda x: fun_and_grad(x), x0, method="BFGS", jac=True,
                   options={"maxiter": 10000, "disp": True, "gtol": 1.0e-4})

    x_opt = _project_linked(sol.x)

    if l_axial_sym:
        symmetrize_xorbrot(list_orb_rot, x_opt, list_degmo)

    obt_opt, tbt_opt = orthogonal_transform_obt_tbt(
        x_opt, list_orb_rot, obt_spatial, tbt_spatial
    )

    Hmat_opt = construct_Hmat_CSFs_paral_triu(list_CSF, Enuc, obt_opt, tbt_opt, nparal)
    evals = get_lowest_n_eigen(Hmat_opt, n_roots)

    weights_arr = np.array(weights, dtype=float)
    Eavg = float(np.dot(weights_arr, evals))

    print(f"\nWeighted energy sum: {Eavg}")
    print(f"Individual eigenvalues: {evals}")

    return x_opt, Eavg, obt_opt, tbt_opt

def orthogonal_transform_obt_tbt(x_orbrot,list_orb_rot,obt_spatial,tbt_phys_spatial):
    """
    Given the orbital rotational angles and orbital pairs, transform the obt and tbt of spatial orbitals
    """

    assert len(x_orbrot) == len(list_orb_rot)

    n_spatialmo = obt_spatial.shape[0]
    kappa_mat = np.zeros([n_spatialmo,n_spatialmo])

    for ix, pair in enumerate(list_orb_rot):
        [iorb,jorb] = pair
        kappa_mat[iorb,jorb] = x_orbrot[ix]
        kappa_mat[jorb,iorb] = -x_orbrot[ix]


    Omat = scipy.linalg.expm(kappa_mat)
   #print('\nkappa_mat')
   #print_matrix(kappa_mat)
   #print('\nOmat:')
   #print_matrix(Omat)

    obt_phys_spatial_trans = np.einsum('pq,pa,qb->ab',obt_spatial,Omat,Omat,optimize=True)
    tbt_phys_spatial_trans = np.einsum('pqrs,pa,qb,rc,sd->abcd',tbt_phys_spatial,Omat,Omat,Omat,Omat,optimize=True)

    obt_phys_spin_trans = obt_phys_spatial_to_spin(obt_phys_spatial_trans)
    tbt_phys_spin_trans = tbt_phys_spatial_to_spin(tbt_phys_spatial_trans)

    return obt_phys_spin_trans, tbt_phys_spin_trans

def construct_Hmat_CSFs(list_CSF,Enuc,obt,tbt,lrdm=False):
    list_list_rdm1 = []
    list_list_rdm2 = []
    ndim_CSF = len(list_CSF)
    Hmat_CSF = np.zeros([ndim_CSF,ndim_CSF])
    for iCSF in range(ndim_CSF):
        CSFi = list_CSF[iCSF]
        for jCSF in range(iCSF,ndim_CSF):
            CSFj = list_CSF[jCSF]
            if lrdm:
                Helm, list_rdm1, list_rdm2 = Helm_between_CSFs(Enuc,obt,tbt,CSFi,CSFj,lrdm)
                list_list_rdm1.append(list_rdm1)
                list_list_rdm2.append(list_rdm2)
            else:
                Helm = Helm_between_CSFs(Enuc,obt,tbt,CSFi,CSFj)
            Selm = overlap_CSFs(CSFi,CSFj)
            l_orthonormal = False
            if iCSF == jCSF and np.isclose(Selm,1.0): l_orthonormal = True
            if iCSF != jCSF and np.isclose(Selm,0.0): l_orthonormal = True
            if not l_orthonormal:
                print(f'Non-orthonormal CSFs {iCSF,jCSF,Selm}. Bombing out!')
                print('CSFi')
                print(CSFi)
                print('CSFj')
                print(CSFj)
                sys.exit()
            Hmat_CSF[iCSF,jCSF] = Helm
            Hmat_CSF[jCSF,iCSF] = Helm

    if lrdm:
        return Hmat_CSF, list_list_rdm1, list_list_rdm2
    else:
        return Hmat_CSF

def construct_Hmat_CSFs_paral_triu_old(list_CSF,Enuc,obt,tbt,nparal=1,lrdm=False):

    ndim_CSF = len(list_CSF)
    Hmat_CSF = np.zeros([ndim_CSF,ndim_CSF])

    i_upper = np.triu_indices(ndim_CSF)

   #assert ndim_CSF*(ndim_CSF+1) / 2 == len(i_upper[0])

    i_upper_list = []
    for i in range(len(i_upper[0])):
       #print(i_upper[0][i],i_upper[1][i])
        i_upper_list.append([i_upper[0][i],i_upper[1][i]])

    n_all_pairs = len(i_upper[0])
   #chunk_size = n_all_pairs // nparal
   #print(f'# of all pairs: {n_all_pairs}, divided by nparal: {nparal}')
   #print(f'# of bra-ket pairs handled by each core = {chunk_size}')

   #chunks = []
   #for ii in range(0,nparal):
   #    chunks.append(i_upper_list[ii*chunk_size:(ii+1)*chunk_size])

   #chunks[-1] += i_upper_list[nparal*chunk_size:]

    index_chunks = even_distrib_ncomp_to_nparal(n_all_pairs,nparal)
    nchunk = len(index_chunks)

   #list_list_outcome = Parallel(n_jobs=nparal)(delayed(Helm_for_one_list_pairs_braket)(chunk,list_CSF,Enuc,obt,tbt,lrdm) for chunk in chunks)
    list_list_CSF = []
    list_Enuc = []
    list_obt = []
    list_tbt = []
    list_lrdm = []
    chunks = []
    for ii in range(nchunk):
        chunks.append(i_upper_list[index_chunks[ii][0]:index_chunks[ii][-1]+1])
        list_list_CSF.append(list_CSF)
        list_Enuc.append(Enuc)
        list_obt.append(obt)
        list_tbt.append(tbt)
        list_lrdm.append(lrdm)
    list_list_outcome = Parallel(n_jobs=nchunk)(delayed(Helm_for_one_list_pairs_braket)(chunks[ii],list_list_CSF[ii],list_Enuc[ii],list_obt[ii],list_tbt[ii],list_lrdm[ii]) for ii in range(nchunk))

   #print(len(list_list_outcome))

    if lrdm:
        list_list_rdm1 = []
        list_list_rdm2 = []

    for ichunk, chunk in enumerate(chunks):
        list_outcome = list_list_outcome[ichunk]
        for ii, ind_braket in enumerate(chunk):
            output = list_outcome[ii]
            if lrdm:
                Hmat_CSF[ind_braket[0],ind_braket[1]] = output[0]
                list_list_rdm1.append(output[1])
                list_list_rdm2.append(output[2])
            else:
                Hmat_CSF[ind_braket[0],ind_braket[1]] = output

    i_lower = np.tril_indices(ndim_CSF,-1)
    Hmat_CSF[i_lower] = Hmat_CSF.T[i_lower]

    if lrdm:
        return Hmat_CSF,list_list_rdm1,list_list_rdm2
    else:
        return Hmat_CSF

def construct_Hmat_CSFs_paral_triu(list_CSF,Enuc,obt,tbt,nparal=1,lrdm=False,debug=False):

    ndim_CSF = len(list_CSF)
    Hmat_CSF = np.zeros([ndim_CSF,ndim_CSF])

    i_upper = np.triu_indices(ndim_CSF)

    n_all_pairs = len(i_upper[0])
    index_chunks = even_distrib_ncomp_to_nparal(n_all_pairs,nparal,l_only_end_points=True)
    nchunk = len(index_chunks)

    list_i_upper_chunk = []
    list_list_CSF = []
    list_Enuc = []
    list_obt = []
    list_tbt = []
    list_lrdm = []
    for ichunk in range(nchunk):
        [istart, iend] = index_chunks[ichunk]
        i_upper_chunk = (i_upper[0][istart:iend+1],i_upper[1][istart:iend+1])
        list_i_upper_chunk.append(i_upper_chunk)
        list_list_CSF.append(list_CSF)
        list_Enuc.append(Enuc)
        list_obt.append(obt)
        list_tbt.append(tbt)
        list_lrdm.append(lrdm)

    def Hmat_for_chunk_i_upper(i_upper_chunk,list_CSF,Enuc,obt,tbt,lrdm):
        Hmat = lil_matrix((len(list_CSF),len(list_CSF)))
        if lrdm:
            list_list_rdm1 = []
            list_list_rdm2 = []
        for iibra,ibra in enumerate(i_upper_chunk[0]):
            iket = i_upper_chunk[1][iibra]
            CSF_bra, CSF_ket = list_CSF[ibra], list_CSF[iket]
            if lrdm:
                Helm, list_rdm1,list_rdm2 = Helm_between_CSFs(Enuc,obt,tbt,CSF_bra,CSF_ket,lrdm)
                list_list_rdm1.append(list_rdm1)
                list_list_rdm2.append(list_rdm2)
            else:
                Helm = Helm_between_CSFs(Enuc,obt,tbt,CSF_bra,CSF_ket,lrdm)
            Hmat[ibra,iket] = Helm
            Hmat[iket,ibra] = Helm

        Hmat = csr_matrix(Hmat)
        if lrdm:
            return Hmat, list_list_rdm1, list_list_rdm2
        else:
            return Hmat

    # NOTE: default (process) backend on purpose. Hmat_for_chunk_i_upper is
    # CPU-bound Python / tiny-array numpy (Helm_between_SDs), which is GIL-bound;
    # a threading backend would serialize it under the GIL. Processes give real
    # parallelism here despite the one-time cost of shipping the CSF list.
    list_chunk_Hmat = Parallel(n_jobs=nchunk)(delayed(Hmat_for_chunk_i_upper)(list_i_upper_chunk[ii],list_list_CSF[ii],list_Enuc[ii],list_obt[ii],list_tbt[ii],list_lrdm[ii]) for ii in range(nchunk))
    if lrdm:
        for ichunk in range(nchunk):
            if ichunk == 0:
                Hmat = list_chunk_Hmat[ichunk][0]
                list_list_rdm1 = list_chunk_Hmat[ichunk][1]
                list_list_rdm2 = list_chunk_Hmat[ichunk][2]
            else:
                Hmat += list_chunk_Hmat[ichunk][0]
                list_list_rdm1 += list_chunk_Hmat[ichunk][1]
                list_list_rdm2 += list_chunk_Hmat[ichunk][2]
    else:
        for ichunk in range(nchunk):
            if ichunk == 0:
                Hmat = list_chunk_Hmat[ichunk]
            else:
                Hmat += list_chunk_Hmat[ichunk]

    Hmat = Hmat.toarray()

    if debug:
        print(f'Debug comparison of Hmat calculateed using different parallel functions')
        if lrdm:
            Hmat_test, list_list_rdm1_test, list_list_rdm2_test = construct_Hmat_CSFs_paral_triu_old(list_CSF,Enuc,obt,tbt,nparal,lrdm)
        else:
            Hmat_test = construct_Hmat_CSFs_paral_triu_old(list_CSF,Enuc,obt,tbt,nparal,lrdm)

        assert np.isclose(np.linalg.norm(Hmat - Hmat_test),0.0)
        if lrdm:
            assert len(list_list_rdm1) == len(list_list_rdm1_test)
            assert len(list_list_rdm1) == len(list_list_rdm2_test)
            assert len(list_list_rdm2) == len(list_list_rdm2_test)
            for ilist in range(len(list_list_rdm1)):
                list_rdm1 = list_list_rdm1[ilist]
                list_rdm2 = list_list_rdm2[ilist]
                list_rdm1_test = list_list_rdm1_test[ilist]
                list_rdm2_test = list_list_rdm2_test[ilist]
                assert len(list_rdm1) == len(list_rdm1_test)
                assert len(list_rdm2) == len(list_rdm2_test)
                for irdm in range(len(list_rdm1)): assert list_rdm1[irdm] == list_rdm1_test[irdm]
                for irdm in range(len(list_rdm2)): assert list_rdm2[irdm] == list_rdm2_test[irdm]

        print(f'Debug comparison passed')

    if lrdm:
        return Hmat,list_list_rdm1,list_list_rdm2
    else:
        return Hmat

def Helm_for_one_list_pairs_braket(list_ind_braket,list_CSF,Enuc,obt,tbt,lrdm=False):

    list_output = []
    for ind_braket in list_ind_braket:
        CSFbra, CSFket = list_CSF[ind_braket[0]], list_CSF[ind_braket[1]]

        if lrdm:
            Helm, list_rdm1, list_rdm2 = Helm_between_CSFs(Enuc,obt,tbt,CSFbra,CSFket,lrdm)
            list_output.append([Helm,list_rdm1,list_rdm2])
        else:
            Helm = Helm_between_CSFs(Enuc,obt,tbt,CSFbra,CSFket)
            list_output.append(Helm)

    return list_output


def Helm_for_one_pair_braket(ind_braket,list_CSF,Enuc,obt,tbt,lrdm=False):

    CSFbra, CSFket = list_CSF[ind_braket[0]], list_CSF[ind_braket[1]]

    if lrdm:
        Helm, list_rdm1, list_rdm2 = Helm_between_CSFs(Enuc,obt,tbt,CSFbra,CSFket,lrdm)
        return [Helm,list_rdm1,list_rdm2]
    else:
        Helm = Helm_between_CSFs(Enuc,obt,tbt,CSFbra,CSFket)
        return Helm


def construct_Hmat_CSFs_paral(list_CSF,Enuc,obt,tbt,nparal=1,lrdm=False):
   #print(f'In construct_Hmat_CSFs_paral, nparal = {nparal}')

    list_list_rdm1 = []
    list_list_rdm2 = []
    ndim_CSF = len(list_CSF)
    Hmat_CSF = np.zeros([ndim_CSF,ndim_CSF])

    list_vec_Helm_for_one_bra = Parallel(n_jobs=nparal)(delayed(Helm_for_one_CSFbra)(ibra,list_CSF,Enuc,obt,tbt,lrdm) for ibra in range(ndim_CSF))

    if lrdm:
        for i in range(ndim_CSF):
            Hmat_CSF[i,:] = list_vec_Helm_for_one_bra[i][0]
            list_list_rdm1 += list_vec_Helm_for_one_bra[i][1]
            list_list_rdm2 += list_vec_Helm_for_one_bra[i][2]
    else:
        for i in range(ndim_CSF):
            Hmat_CSF[i,:] = list_vec_Helm_for_one_bra[i]

   #Hmat_CSF = Hmat_CSF + Hmat_CSF.T - np.diag(np.diag(Hmat_CSF))
    i_lower = np.tril_indices(ndim_CSF,-1)
    Hmat_CSF[i_lower] = Hmat_CSF.T[i_lower]


    if lrdm:
        return Hmat_CSF, list_list_rdm1,list_list_rdm2
    else:
        return Hmat_CSF

def Helm_for_one_CSFbra(iCSFbra,list_CSF,Enuc,obt,tbt,lrdm=False):

    CSFi = list_CSF[iCSFbra]
    ndim = len(list_CSF)
    vec_Helm_one_bra = np.zeros([ndim])
    list_list_rdm1 = []
    list_list_rdm2 = []
    for jCSFket in range(iCSFbra,ndim):
        CSFj = list_CSF[jCSFket]
        if lrdm:
            Helm, list_rdm1, list_rdm2 = Helm_between_CSFs(Enuc,obt,tbt,CSFi,CSFj,lrdm)
            vec_Helm_one_bra[jCSFket] = Helm
            list_list_rdm1.append(list_rdm1)
            list_list_rdm2.append(list_rdm2)
        else:
            Helm = Helm_between_CSFs(Enuc,obt,tbt,CSFi,CSFj)
            vec_Helm_one_bra[jCSFket] = Helm


    if lrdm:
        return [vec_Helm_one_bra,list_list_rdm1,list_list_rdm2]
    else:
        return vec_Helm_one_bra


def remove_zero_rdm_elements(list_rdm,small=1.0e-8):
    """
    For read-in list of reduced density matrix elements, only return the nonzero ones
    """

    ndim = len(list_rdm)
    list_l_remove = [False] * ndim
    for ii, item in enumerate(list_rdm):
        if abs(item[-1]) < small: list_l_remove[ii] = True

    for ii in range(ndim-1,-1,-1):
        if list_l_remove[ii]: del list_rdm[ii]

def check_orthonormal_CSFs(list_CSF):
    ndim_CSF = len(list_CSF)
    for iCSF in range(ndim_CSF):
        CSFi = list_CSF[iCSF]
        for jCSF in range(iCSF,ndim_CSF):
            CSFj = list_CSF[jCSF]
            Selm = overlap_CSFs(CSFi,CSFj)
            l_orthonormal = False
            if iCSF == jCSF and np.isclose(Selm,1.0): l_orthonormal = True
            if iCSF != jCSF and np.isclose(Selm,0.0): l_orthonormal = True
            if not l_orthonormal:
                print(f'Non-orthonormal CSFs {iCSF,jCSF,Selm}. Bombing out!')
                print('CSFi:')
                print(CSFi)
                print('CSFj:')
                print(CSFj)
                sys.exit()

def op_to_create_hp(onvec,debug=False):
    """
    Read in a ON vector and return the excitation operator that makes this ON from
    a reference with all lowest-indices spin orbitals being occupied
    """

    occ_spinorb = np.where(onvec == 1.0)[0]
    unocc_spinorb = np.where(onvec == 0.0)[0]
    nelec = len(occ_spinorb)
    hoso = nelec - 1
    if debug:
        print(occ_spinorb,nelec,hoso)
        print(unocc_spinorb)

   #print(np.where(occ_spinorb > hoso))
    part_spinorb = occ_spinorb[np.where(occ_spinorb > hoso)]
    hole_spinorb = unocc_spinorb[np.where(unocc_spinorb <= hoso)]
   #part_spinorb = np.where(occ_spinorb > hoso)[0]
   #hole_spinorb = np.where(unocc_spinorb <= hoso)[0]
    if debug: print(part_spinorb,hole_spinorb)

    assert len(part_spinorb) == len(hole_spinorb)
    op = FermionOperator.identity()
    cur_on = onvec.copy()
    phase = 1.0
    for ihp, hole in enumerate(hole_spinorb):
        part = part_spinorb[ihp]
       #print(np.where(occ_spinorb > hole and occ_spinorb <= hoso))
       #occ_spinorb[occ_spinorb > hole and occ_spinorb <= hoso]
       #nflip = len(occ_spinorb[np.where((occ_spinorb > hole) & (occ_spinorb <= hoso))])
        nflip = sum(cur_on[:part])
        cur_on[part] = 0.0
        nflip += sum(cur_on[:hole])
        cur_on[hole] = 1.0
        phase *= (-1.0)**nflip
        if debug: print(part,hole,nflip,phase)
        term = ((int(part),1),(int(hole),0))
        op *= FermionOperator(term,1.0)

    op *= phase
    op = normal_ordered(op)
    if debug:
        print('\nExcitation operator that creates the read in ON vec:')
        print(op)

    return op

#def screen_CSFs_chain_Helm_with_CSF0(list_CSF,Enuc,obt,tbt,nparal=1,small=1e-4,debug=False):
    """
    Select CSFs based on chain of matrix elements with CSF0.
    Solve the eigenvalue problem of the Hamiltonian matrix of list_CSF
    basis set. Select the eigenstate with the lowest eigenvalue and with
    non-zero amplitude of CSF0. All CSFs that have nonzero amplitudes
    in this state are selected.
    """

    if debug: print('\nIn screen_CSFs_chain_Helm_with_CSF0')

    if nparal > 1:
        Hmat = construct_Hmat_CSFs_paral_triu(list_CSF, Enuc, obt, tbt, nparal)
    else:
        Hmat = construct_Hmat_CSFs(list_CSF, Enuc, obt, tbt)
    ndim = len(list_CSF)
    # eigval,eigvec = np.linalg.eigh(Hmat)
    nstate = min(10, ndim)
    if nstate == ndim: nstate = nstate // 2
    Hmat_sparse = csr_matrix(Hmat)
    eigval, eigvec = scipy.sparse.linalg.eigsh(Hmat_sparse, k=nstate, which='SA')
    if debug:
        print('\nEigensolution all CSFs')
        print_eigen_solution(eigval[:nstate], eigvec[:, :nstate])

    iCSF_interest = 0
    for istate in range(ndim):
        if abs(eigvec[iCSF_interest, istate]) > small:
            break

    if debug:
        print(f'The lowest state with nonzero amplitude of CSF{iCSF_interest} is State {istate}')

    list_l_remove_CSF = [False] * ndim
    n_to_remove = 0
    for iCSF in range(ndim):
        if abs(eigvec[iCSF, istate]) < small:
            # print(f'To remove {iCSF} with {eigvec[iCSF,istate]} amplitude')
            list_l_remove_CSF[iCSF] = True
            n_to_remove += 1

    print(f'# of CSFs to be removed: {n_to_remove}')
    return list_l_remove_CSF

from functools import reduce

def csf_symmetry_from_somos(somo_tuple, dmo_tuple, mo_sym, c2v_mult):
    """
    somo_tuple: tuple of MO labels (strings like '0','4',...)
    dmo_tuple: tuple of MO labels (strings like '0','4',...)
    returns: total C2v irrep label (e.g. 'A1')
    """
    somo_irreps = [mo_sym[mo] for mo in somo_tuple]
    dmo_irreps = [mo_sym[mo] for mo in dmo_tuple]
    irreps = somo_irreps + dmo_irreps + dmo_irreps

    if not irreps:
        return 'A1'

    return reduce(lambda a, b: c2v_mult[(a, b)], irreps)

def screen_CSFs_multi_ref_chain(list_CSF, ndim, Enuc, obt, tbt,nparal=1, n=2, small=1e-4, debug=False):

    if debug:
        print("\nIn screen_CSFs_multi_ref_chain")

    if nparal > 1:
        Hmat = construct_Hmat_CSFs_paral_triu(list_CSF, Enuc, obt, tbt, nparal)
    else:
        Hmat = construct_Hmat_CSFs(list_CSF, Enuc, obt, tbt)

    if ndim < 2:
        return [False] * ndim

    k = min(max(n, 1), ndim - 1)
    Hmat_sparse = csr_matrix(Hmat)
    eigval, eigvec = scipy.sparse.linalg.eigsh(Hmat_sparse, k=k, which="SA")

    # sort eigenpairs
    idx = np.argsort(eigval)
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]

    diagH = np.diag(Hmat)
    mu_sorted = np.argsort(diagH)
    M = mu_sorted[:n]

    if debug:
        print("Reference CSFs (lowest diagonal):", M)

    selected_states = []
    used_states = set()

    for mu in M:
        best_k = None
        best_amp = 0.0

        for k_idx in range(eigvec.shape[1]):
            if k_idx in used_states:
                continue

            amp = abs(eigvec[mu, k_idx])

            if amp > small and amp > best_amp:
                best_amp = amp
                best_k = k_idx

        if best_k is not None:
            selected_states.append(best_k)
            used_states.add(best_k)

        if debug:
            print(f"CSF {mu} -> state {best_k} (amp={best_amp})")

    if len(selected_states) == 0:
        return [False] * ndim

    selected_eigvec = eigvec[:, selected_states]  # shape (ndim, n_selected)

    max_amp = np.max(np.abs(selected_eigvec), axis=1)
    list_l_remove_CSF = (max_amp < small).tolist()

    if debug:
        print("Selected eigenstates:", selected_states)
        print("# removed:", np.sum(list_l_remove_CSF))

    return list_l_remove_CSF

def screen_CSFs_by_n_lowest_states(list_CSF, ndim, Enuc, obt, tbt, nparal=1, n=2, small=1e-4, debug=False):
    """
    Multi-root CSF screening.

    Build the Hamiltonian in the CSF basis, compute the n lowest-energy eigenstates,
    and remove a CSF only if it is "useless for all" of those states, i.e. its
    coefficient magnitude is below `small` in every one of the n eigenvectors.

    Criterion:
        remove CSF i  iff  max_{s=0..n-1} |c_i^(s)| < small

    Returns:
        list_l_remove_CSF : list[bool] of length ndim (True => remove)
    """

    if debug:
        print("\nIn screen_CSFs_chain_Helm_with_CSF0 (multi-root)")

    # --- Build Hamiltonian ---
    if nparal > 1:
        Hmat = construct_Hmat_CSFs_paral_triu(list_CSF, Enuc, obt, tbt, nparal)
    else:
        Hmat = construct_Hmat_CSFs(list_CSF, Enuc, obt, tbt)

    if ndim != Hmat.shape[0]:
        raise ValueError(f"ndim={ndim} but Hamiltonian dimension is {Hmat.shape[0]}")

    # --- Choose how many eigenpairs to compute ---
    # eigsh requires: 0 < k < ndim
    if ndim < 2:
        # nothing meaningful to screen
        return [False] * ndim

    k = min(n, ndim - 1)  # ensure k < ndim

    # --- Solve for k lowest eigenpairs ---
    Hmat_sparse = csr_matrix(Hmat)
    eigval, eigvec = scipy.sparse.linalg.eigsh(Hmat_sparse, k=k, which="SA")

    # eigsh doesn't guarantee sorted eigenvalues
    idx = np.argsort(eigval)
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]

    if debug:
        print(f"\nComputed {k} lowest eigenpairs (which='SA'):")
        print_eigen_solution(eigval, eigvec)

    # --- (Optional) print the "CSF0-like" diagnostic used in the old function ---
    diagH = np.diag(Hmat)
    iCSF_interest = int(np.argmin(diagH))
    print("Lowest <CSF|H|CSF> is CSF", iCSF_interest, "value =", diagH[iCSF_interest])

    if debug:
        overlaps = np.abs(eigvec[iCSF_interest, :])
        print(f"Abs(coeff) of CSF {iCSF_interest} in the {k} lowest states:", overlaps)

    # --- Multi-root screening: remove only if small in ALL states ---
    max_amp = np.max(np.abs(eigvec), axis=1)  # shape (ndim,)
    list_l_remove_CSF = (max_amp < small).tolist()
    n_to_remove = int(np.sum(max_amp < small))

    if debug:
        print(f"# of CSFs to be removed (max_amp < {small} across {k} states): {n_to_remove}")

    return list_l_remove_CSF

def construct_list_genmat_from_occ(list_ex_space,list_ia_pairs,debug=False):
    """
    Construct list of generating matrices of Tiiaa^00 based on occupancies.
    Assuming that all CSFs in list_ex_space only differ in dmo
    """

    ndim = len(list_ex_space)
    list_set_dmo = []
    for CSF in list_ex_space:
        list_dmo = dmo_in_SD(CSF[0][0])
        set_dmo = set(list_dmo)
        list_set_dmo.append(set_dmo)

    list_ia_accumul = []
    list_ia_associated_CSFs = []
    for iCSF in range(ndim):
        set_dmo_i = list_set_dmo[iCSF]
        for jCSF in range(iCSF+1,ndim):
            set_dmo_j = list_set_dmo[jCSF]
            set_dmo_diff = set_dmo_i^set_dmo_j
            if len(set_dmo_diff) == 0:
                print(f'Impossible identical dmo sets for CSFs{iCSF} and {jCSF}')
                print(set_dmo_i,set_dmo_j)
                print('Bombing out!')
                sys.exit()
            if len(set_dmo_diff) % 2 == 1:
                print(f'Impossible odd difference between dmo sets for CSFs{iCSF} and {jCSF}')
                print(set_dmo_i,set_dmo_j)
                print('Bombing out!')
                sys.exit()
            if len(set_dmo_diff) > 2: continue
            if list(set_dmo_diff) in list_ia_accumul:
                ia_ind = list_ia_accumul.index(list(set_dmo_diff))
                list_ia_associated_CSFs[ia_ind].append([iCSF,jCSF])
            else:
                list_ia_accumul.append(list(set_dmo_diff))
                list_ia_associated_CSFs.append([[iCSF,jCSF]])

    if debug:
        print('\nlist_set_dmo:')
        for ii,item in enumerate(list_set_dmo):
            print(ii,item)
        for ii, item in enumerate(list_ia_accumul):
            print(f'difference dmo pair: {item}')
            print(list_ia_associated_CSFs[ii])


    list_genmat = []
    for ia_pairs in list_ia_pairs:
        genmat = csr_matrix((ndim,ndim))
        print(f'ia_pairs: {ia_pairs}')
        for pair in ia_pairs:
            pair_reversed = copy.deepcopy(pair)
            pair_reversed.reverse()
           #print(f'pair and reversed: {pair,pair_reversed}')
            if pair not in list_ia_accumul and pair_reversed not in list_ia_accumul:
                print(f'Strange! {pair} or {pair_reversed} is not in list_ia_accumul: {list_ia_accumul}')
                print('Bombing out!')
                sys.exit()

            elif pair in list_ia_accumul:
                pair_ind = list_ia_accumul.index(pair)
                for CSFpair in list_ia_associated_CSFs[pair_ind]:
                    genmat[CSFpair[1],CSFpair[0]] =  1.0
                    genmat[CSFpair[0],CSFpair[1]] = -1.0
            elif pair_reversed in list_ia_accumul:
                pair_ind = list_ia_accumul.index(pair_reversed)
                for CSFpair in list_ia_associated_CSFs[pair_ind]:
                    genmat[CSFpair[1],CSFpair[0]] =  1.0
                    genmat[CSFpair[0],CSFpair[1]] = -1.0
        if debug:
            print('generating matrix:')
            print(genmat)
        list_genmat.append(genmat)

    return list_genmat

def symmetrize_xorbrot(list_orb_rot,x_orbrot,list_degmo,debug=False):
    """
    Rotations of degenerate orbital pairs should have the same rotational angle.
    This function is to perform such a symmetrization.
    """

    same_xrot_thrsh = 1.0e-4
    assert len(list_orb_rot) == len(x_orbrot)
    for ipair in range(len(list_orb_rot)):
        [iorb1,iorb2] = list_orb_rot[ipair]
        x_rot_i = x_orbrot[ipair]
        for jpair in range(ipair+1,len(list_orb_rot)):
            [jorb1,jorb2] = list_orb_rot[jpair]
            x_rot_j = x_orbrot[jpair]
            if ([iorb1,jorb1] in list_degmo or [jorb1,iorb1] in list_degmo) and \
              ([iorb2,jorb2] in list_degmo or [jorb2,iorb2] in list_degmo):
                print(f'Rotations to be symmetrized: {[iorb1,iorb2],[jorb1,jorb2]}')
                if abs(x_rot_i - x_rot_j) > same_xrot_thrsh:
                    print(f'Warning! Original read-in rotational angles differ > {same_xrot_thrsh}')
                x_rot_sym = 0.5*(x_rot_i + x_rot_j)
                x_orbrot[ipair] = x_rot_sym
                x_orbrot[jpair] = x_rot_sym

def BCH_one_G_one_Hterm(H,G,theta):
    """
    e^(-theta*G) H e^(theta*G). H and TG are strings of 2nd quantized operators.
    G = TG - TG^+
    """

    ad1 = normal_ordered(commutator(H,G))
    ad1.compress()
    if FermionOperator.isclose(ad1,FermionOperator.zero()):
        return H

    ad2 = normal_ordered(commutator(ad1,G))
    ad2.compress()
    ad3 = normal_ordered(commutator(ad2,G))
    ad3.compress()
    test1 = normal_ordered(ad3 + 1.0*ad1)
    test1.compress()
    if FermionOperator.isclose(test1,FermionOperator.zero()):
        H_new  = H + np.sin(theta)*ad1
        H_new += (1.0 - np.cos(theta))*ad2
        H_new = normal_ordered(H_new)
        H_new.compress()
        return H_new

    test4 = normal_ordered(ad3 + 4.0*ad1)
    test4.compress()
    if FermionOperator.isclose(test4,FermionOperator.zero()):
        H_new = H + 0.5*np.sin(2.0*theta)*ad1
        H_new += 0.5*((np.sin(theta))**2.0)*ad2
        H_new = normal_ordered(H_new)
        H_new.compress()
        return H_new

    print(f'Both test4 and test1 fail. This should not happen')
    print('term:')
    print(term)
    print('test4:')
    print(test4)
    print('test1:')
    print(test1)
    sys.exit()

def remove_terms_inactive_virtual(Op,actmo_start,actmo_end,n_spatialmo,nparal=1):
    """
    Remove the following terms
    term*(1-n) = 0, (1-n)*term = 0 for virtual orbitals
    term*n = 0, n*term = 0 for inactive orbitals
    """

    n_spinmo = 2*n_spatialmo
    n_term_before = len(Op.terms)
   #list_Op_terms = list(Op.terms)
   #list_l_remove = False*len(list_Op_terms)
   #for iterm,term in enumerate(list_Op_terms):
   #    print(iterm,term)
    Op_new = Op
    nterm_old = len(list(Op_new.terms))
    for inactmo in range(0,actmo_start):
        print(f'Removing inacmo: {inactmo}')
        nop_alpha = number_operator(n_spinmo,inactmo*2)
        nop_beta  = number_operator(n_spinmo,inactmo*2+1)
        proj_op = normal_ordered(nop_alpha*nop_beta)
        proj_op.compress()
       #Op = Op*nop_alpha*nop_beta
       #Op = normal_ordered(Op)
       #Op.compress()
       #Op = nop_alpha*nop_beta*Op
       #Op = normal_ordered(Op)
       #Op.compress()
        list_new_terms = Parallel(n_jobs = nparal)(delayed(triple_product_op)(proj_op,term,proj_op) for term in Op_new)
        Op_new = FermionOperator.zero()
        for term in list_new_terms:
            Op_new += term
        Op_new.compress()
        nterm_new = len(list(Op_new.terms))
        print(f'Term reduction from {nterm_old} to {nterm_new} by {nterm_old-nterm_new}')
        nterm_old = nterm_new

    n_term_after_inact = len(Op_new.terms)
    print(f'Number of terms reduced from {n_term_before} to {n_term_after_inact}, by {n_term_before-n_term_after_inact}')

    for virmo in range(actmo_end+1,n_spatialmo):
        print(f'Removing virmo: {virmo}')
        nop_alpha = number_operator(n_spinmo,virmo*2)
        nop_beta  = number_operator(n_spinmo,virmo*2+1)
        proj_op = normal_ordered((1.0-nop_alpha)*(1.0-nop_beta))
        proj_op.compress()
       #Op = Op*(1.0-nop_alpha)*(1.0-nop_beta)
       #Op = normal_ordered(Op)
       #Op.compress()
       #Op = (1.0-nop_alpha)*(1.0-nop_beta)*Op
       #Op = normal_ordered(Op)
       #Op.compress()
        list_new_terms = Parallel(n_jobs = nparal)(delayed(triple_product_op)(proj_op,term,proj_op) for term in Op_new)
        Op_new = FermionOperator.zero()
        for term in list_new_terms:
            Op_new += term
        Op_new.compress()
        nterm_new = len(list(Op_new.terms))
        print(f'Term reduction from {nterm_old} to {nterm_new} by {nterm_old-nterm_new}')
        nterm_old = nterm_new

    n_term_after_vir = len(Op_new.terms)
    print(f'Number of terms reduced from {n_term_after_inact} to {n_term_after_vir}, by {n_term_after_inact-n_term_after_vir}')

def triple_product_op(A,B,C):

    res = normal_ordered(A*B*C)
    res.compress()
    return res

def remove_Hterms_violating_seniority(Op,actmo_start,actmo_end,n_spatialmo):

    n_spinmo = n_spatialmo*2
    Op_old = Op
    list_mo_omega0 = []
    for mo in range(actmo_start):
        list_mo_omega0.append(mo)
    for mo in range(actmo_end+1,n_spatialmo):
        list_mo_omega0.append(mo)
    for mo in list_mo_omega0:
        print(f'Screening mo {mo}')
        nop_alpha = number_operator(n_spinmo,mo*2)
        nop_beta  = number_operator(n_spinmo,mo*2+1)
        proj_op  = nop_alpha*nop_beta
        proj_op += (1.0 - nop_alpha)*(1.0 - nop_beta)
        list_screened_terms = []
        for term in Op_old:
            test_op = triple_product_op(proj_op,term,proj_op)
            if FermionOperator.isclose(test_op,FermionOperator.zero()):
               #print(f'Removing {term}')
                screened_term = FermionOperator.zero()
            else:
                screened_term = term
            list_screened_terms.append(screened_term)

        Op_new = FermionOperator.zero()
        for term in list_screened_terms:
            Op_new += term

        Op_new.compress()
        print(f'# of terms reduced from {len(Op_old.terms)} to {len(Op_new.terms)}')
        Op_old = Op_new

    return Op_new

def proj_Tiiaa_plus_Tiibb(i,a,b,debug=False):
    """
    Read in the three spatial orbital indices and give the projection operator for eigenvalues of
    i and -i for 1/sqrt(Tiiaa + Tiibb)
    P = 1/2(n_ia*n_ib + (1-n_ia)*(1-n_ib))*
        (n_aa*n_ab + (1-n_ba)*(1-n_bb) + n_ba*n_bb + (1-n_aa)*(1-n_ab)
         + a_aa^+ a_ab^+ a_bb a_ba + a_ba^+ a_bb^+ a_ab a_aa)
        + n_ia*n_ib*(1-n_aa)*(1-n_ab)*(1-n_ba)*(1-n_bb)
        + (1-n_ia)*(1-n_ib)*n_aa*n_ab*n_ba*n_bb
    """

    if debug: print(f'\n In proj_Tiiaa_plus_Tiibb')

    n_ia = number_ferm_tz(2*i)
    n_ib = number_ferm_tz(2*i+1)
    one_m_n_ia = number_ferm_tz(2*i,True)
    one_m_n_ib = number_ferm_tz(2*i+1,True)
    n_aa = number_ferm_tz(2*a)
    n_ab = number_ferm_tz(2*a+1)
    one_m_n_aa = number_ferm_tz(2*a,True)
    one_m_n_ab = number_ferm_tz(2*a+1,True)
    n_ba = number_ferm_tz(2*b)
    n_bb = number_ferm_tz(2*b+1)
    one_m_n_ba = number_ferm_tz(2*b,True)
    one_m_n_bb = number_ferm_tz(2*b+1,True)



    Ladder = ((2*a,1),(2*a+1,1),(2*b+1,0),(2*b,0))
    Ladder = FermionOperator(Ladder, 1.0)
    Ladder += hermitian_conjugated(Ladder)

    P2eab = n_aa*n_ab*one_m_n_ba*one_m_n_bb \
           +n_ba*n_bb*one_m_n_aa*one_m_n_ab

    P2eab += Ladder
    P2eab *= 0.5

    P2eab = normal_ordered(P2eab)
    P2eab.compress()
    assert check_idempotent(P2eab)

    P2e = n_ia*n_ib*one_m_n_aa*one_m_n_ab*one_m_n_ba*one_m_n_bb \
         +one_m_n_ia*one_m_n_ib*P2eab

    P2e = normal_ordered(P2e)
    P2e.compress()
    assert check_idempotent(P2e)

    P4e = one_m_n_ia*one_m_n_ib*n_aa*n_ab*n_ba*n_bb \
         +n_ia*n_ib*P2eab

    P4e = normal_ordered(P4e)
    P4e.compress()
    assert check_idempotent(P4e)

    P3eup = n_ia*n_ib*one_m_n_ab*one_m_n_bb*(n_aa*one_m_n_ba + n_ba*one_m_n_aa) \
           +one_m_n_ia*one_m_n_ib*n_aa*n_ba*(one_m_n_ab*n_bb+n_ab*one_m_n_bb)

    P3eup = normal_ordered(P3eup)
    P3eup.compress()
    assert check_idempotent(P3eup)

    P3edn = n_ia*n_ib*one_m_n_aa*one_m_n_ba*(n_ab*one_m_n_bb + n_bb*one_m_n_ab) \
           +one_m_n_ia*one_m_n_ib*n_ab*n_bb*(one_m_n_aa*n_ba+n_aa*one_m_n_ba)

    P3edn = normal_ordered(P3edn)
    P3edn.compress()
    assert check_idempotent(P3edn)

    if debug:
        assert check_zero(P2e*P4e)
        assert check_zero(P2e*P3eup)
        assert check_zero(P2e*P3edn)
        assert check_zero(P4e*P3eup)
        assert check_zero(P4e*P3edn)
        assert check_zero(P3eup*P3edn)

    Proj = P2e + P4e + P3eup + P3edn
    Proj = normal_ordered(Proj)
    Proj.compress()
    assert check_idempotent(Proj)


    return Proj, P2e,P4e,P3eup,P3edn

def check_idempotent(Op):
    """
    Check idempotent of input operator
    """

    residue = Op*Op - Op
    residue = normal_ordered(residue)
    residue.compress()
    return FermionOperator.isclose(residue,FermionOperator.zero())
   #if residue == FermionOperator.zero():
   #    return True
   #else:
   #    return False

def check_zero(Op):
    """
    Check Op1*Op2 = zero
    """

    Op = normal_ordered(Op)
    Op.compress()

    return FermionOperator.isclose(Op,FermionOperator.zero())

def even_distrib_ncomp_to_nparal(ncomp,nparal,l_only_end_points=False,debug=False):
    """
    Evenly distribute ncomp components' indices to nparal chunks, in ascending order
    and return a list of the chunks
    """

    chunk_size = ncomp // nparal
    nleft = ncomp % nparal

    chunks = []
    istart = 0
    for ichunk in range(nleft):
        chunk = []
        iend = istart + chunk_size+1
        if l_only_end_points:
            chunk.append(istart)
            chunk.append(iend-1)
        else:
            for ii in range(istart,iend): chunk.append(ii)
        istart = iend
        chunks.append(chunk)
    if chunk_size > 0:
        for ichunk in range(nleft,nparal):
            chunk = []
            iend = istart + chunk_size
            if l_only_end_points:
                chunk.append(istart)
                chunk.append(iend-1)
            else:
                for ii in range(istart,iend): chunk.append(ii)
            istart = iend
            chunks.append(chunk)

    if debug and l_only_end_points:
        print(f'End points of {len(chunks)} chunks:')
        for item in chunks: print(item)


    return chunks

def op_action_tz_CSF_in_chunks(op,CSF_input,nparal=1,debug=False):

    n_comp = len(CSF_input[0])
    chunks = even_distrib_ncomp_to_nparal(n_comp,nparal)
    nchunk = len(chunks)
    assert nchunk <= nparal
   #print(f'n_comp: {n_comp}, nparal: {nparal}, nchunk: {nchunk} in op_action_tz_CSF_in_chunks')

    list_CSF_input = []
    list_op = []
    for chunk in chunks:
        list_CSF_input.append([CSF_input[0][chunk[0]:chunk[-1]+1],\
                         CSF_input[1][chunk[0]:chunk[-1]+1],\
                         CSF_input[2][chunk[0]:chunk[-1]+1]])
        list_op.append(op)

   #list_CSF_output = []
   #for ii in range(nchunk):
   #    list_CSF_output.append(op_action_tz_CSF(op,list_CSF_input[ii]))

    tic = time.perf_counter()
    list_CSF_output = Parallel(n_jobs=nchunk)(delayed(op_action_tz_CSF)(list_op[ii],list_CSF_input[ii]) for ii in range(nchunk))
    toc = time.perf_counter()
    print(f'Time for parallel operator action: {toc - tic}')

    tic = time.perf_counter()
    LC_coefs = np.ones([nchunk])
    resCSF = LC_CSFs(list_CSF_output,LC_coefs)
    toc = time.perf_counter()
    print(f'Time for LC_CSFs: {toc - tic}')

    if debug:
        resCSF_check = op_action_tz_CSF(op,CSF_input)
        assert identical_CSFs(resCSF,resCSF_check)

    return resCSF
