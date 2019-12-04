from math import pi
from math import sqrt
import numpy as np
from numpy import linalg

import itertools
from functools import reduce
import operator
from femutils import GaussPoint

class SpectralElement(object):
    def __init__(self, NGL):
        """Constructor of the SpElem2D class."""
        # NGL: Number of nodes in one direction. This is expanded in "nnode"
        # to the total number of nodes
        self.NGL = NGL
        self.nnode = NGL ** 2
        self.nnodedge = NGL - 2
        self.nnodcell = (NGL - 2) ** 2
        self.dim = 2
        self.elemType = 'Spectral2D(%d)' % NGL

        # First, create points in 1D
        nodes1D, operWei = self._lobattoPoints(NGL)
        gps1D, fullWei = self._gaussPoints(NGL) if NGL <= 3 else \
            self._lobattoPoints(NGL)
        gps_red1D, redWei = self._gaussPoints(NGL - 1)
        cnodes1D, _ = self._lobattoPoints(2)

        # Second, call the "interpFun2D" method to build the Gauss points,
        # the shape functions matrix and the matrix with its derivatives
        (self.H, self.Hrs, self.gps) = \
            self.interpFun2D(nodes1D, gps1D, fullWei)

        (self.HRed, self.HrsRed, self.gpsRed) = \
            self.interpFun2D(nodes1D, gps_red1D, redWei)

        (self.HOp, self.HrsOp, self.gpsOp) = \
            self.interpFun2D(nodes1D, nodes1D, operWei)

        (self.HCoo, self.HrsCoo, self.gpsCoo) = \
            self.interpFun2D(cnodes1D, gps1D, fullWei)

        (self.HCooRed, self.HrsCooRed, self.gpsCooRed) = \
            self.interpFun2D(cnodes1D, gps_red1D, redWei)

        (self.HCooOp, self.HrsCooOp, self.gpsCooOp) = \
            self.interpFun2D(cnodes1D, nodes1D, operWei)

        (self.HCoo1D, _) = self.interpFun1D(cnodes1D, nodes1D)

    # FIXME: We need another interpFun2D for points in 2D with arbitrary (r,s)
    def interpFun2D(self, nodes1D, gps1D, gps1Dwei):
        """Interpolate functions in 2D."""
        (h1D, dh1D) = self.interpFun1D(nodes1D, gps1D)
        nNodes = len(nodes1D)
        ngps = len(gps1D)

        # Reorder polinomial roots
        invPerm = self.getOrder(nNodes)
        # Reorder evaluation points
        invPerm2 = self.getOrder(ngps)

        # Interpolation functions H
        H = list()
        for doubleTern in itertools.product(h1D, h1D):
            auxRow = [reduce(operator.mul, twoPle, 1) for twoPle
                      in itertools.product(*doubleTern)]
            H.append(np.mat([auxRow[y] for y in invPerm]))

        # Derivatives of H wrt R & S
        Hrs = list()
        for doubleTern in itertools.product(dh1D, h1D):
            auxRow = [reduce(operator.mul, twoPle, 1) for twoPle
                      in itertools.product(*doubleTern)]
            Hrs.append(np.mat([[auxRow[y] for y in invPerm],
                               [0]*len(invPerm)]))

        for ind, doubleTern in enumerate(itertools.product(h1D, dh1D)):
            auxRow = [reduce(operator.mul, twoPle, 1) for twoPle
                      in itertools.product(*doubleTern)]
            Hrs[ind][1, :] = [auxRow[y] for y in invPerm]

        # Gauss points
        gps = list()
        for c1 in range(len(gps1D)):
            for c2 in range(len(gps1D)):
                gps.append(GaussPoint(gps1D[c1], gps1D[c2],
                           gps1Dwei[c1] * gps1Dwei[c2]))

        H = [H[i] for i in invPerm2]
        Hrs = [Hrs[i] for i in invPerm2]
        gps = [gps[i] for i in invPerm2]

        return (H, Hrs, gps)

    def interpFun1D(self, Nodes, evalPoi):
        """Interpolate functions in 1D."""
        nevPoi = len(evalPoi)
        nNodes = len(Nodes)

        hFun = np.zeros((nevPoi, nNodes))
        dhFun = np.zeros((nevPoi, nNodes))

        Pat = np.ones((nNodes, nNodes))
        np.fill_diagonal(Pat, 0)

        for ievPoi in range(nevPoi):
            Num1 = evalPoi[ievPoi] * np.ones((nNodes, nNodes))
            Num2 = np.ones((nNodes, 1))*Nodes

            Num = Num1 - Num2
            np.fill_diagonal(Num, 1)
            prodNum = np.prod(Num, axis=1)

            Den = - Num2 + Num2.T
            np.fill_diagonal(Den, 1)

            prodDen = np.prod(Den, axis=1)

            Num3 = np.zeros((nNodes))
            for icol in range(nNodes):
                Num4 = Num.copy()
                Num4[:, icol] = Pat[:, icol]
                Num3 = Num3 + np.prod(Num4, axis=1)

            hFun[ievPoi, :] = prodNum / prodDen
            dhFun[ievPoi, :] = Num3 / prodDen
        return (hFun, dhFun)

    def getOrder(self, nPoints):
        if nPoints > 1:
            Ind = np.zeros((nPoints, nPoints), dtype=int)
            Ind[np.ix_([0, -1], [0, -1])] = np.array([[2, 1], [3, 4]])

            if nNodes > 2:
                Ind[np.ix_([0], range(1, nPoints-1))] = \
                    np.array([range(5 + nPoints - 3, 4, -1)])
                Ind[np.ix_(range(1, nPoints - 1), [0])] = \
                    np.array([range(5 + nPoints - 2, 2 * nPoints + 1)]).T
                Ind[np.ix_([nPoints - 1], range(1, nPoints - 1))] = \
                    np.array([range(2 * nPoints + 1, 3 * nPoints - 1)])
                Ind[np.ix_(range(1, nPoints - 1), [nPoints - 1])] = \
                    np.array([range(4 * nPoints - 4, 3 * nPoints - 2, -1)]).T
                Ind[np.ix_(range(1, nPoints - 1), range(1, nPoints - 1))] = \
                    np.arange(4 * nPoints - 3, nPoints ** 2 + 1).reshape(
                    nPoints - 2, nPoints - 2).T
            Ind -= 1

            Permlst = Ind[::-1].T.reshape(1, np.prod(Ind.shape))[0].tolist()
        else:
            Permlst = [0]

        return [Permlst.index(val) for val in range(len(Permlst))]

    def _gaussPoints(self, N):
        """Calculate the properties of the Gauss points in 1D.

        :param N: Number of Gauss points in 1D.
        :type N: int
        :returns: Tuple with position of the points and their weight.
        :rtype: tuple

        """
        beta = 0.5 / np.sqrt(1.0 - (2.0 * np.arange(1, N)) ** -2)
        T = np.diag(beta, 1) + np.diag(beta, -1)
        [W, V] = linalg.eig(T)
        i = np.argsort(W)
        x = W[i]
        w = 2 * V[0, i] ** 2
        # Symmetrize Gauss points
        x = (x - x[::-1]) / 2
        w = (w + w[::-1]) / 2
        return (x, w)

    def _lobattoPoints(self, N):
        """Calculate the properties of the Gauss-Lobatto points in 1D.

        :param N: Number of Gauss-Lobatto points in 1D.
        :type N: int
        :returns: Tuple with position of the points and their weight.
        :rtype: tuple

        """
        # Use the Chebyshev-Gauss-Lobatto nodes as the first guess
        x = np.cos(np.linspace(0, pi, N))
        # The Legendre Vandermonde Matrix
        P = np.zeros((N, N))
        # Compute P_(N) using the recursion relation
        # Compute its first and second derivatives and
        # update x using the Newton-Raphson method.
        xold = 2
        while np.amax(np.abs(x - xold)) > (1e-15):
            xold = x
            P[:, 0] = 1
            P[:, 1] = x
            for k in range(2, N):
                P[:, k] = ((2 * k - 1) * x * P[:, k - 1] -
                           (k - 1) * P[:, k - 2]) / k
            x = xold - (x * P[:, N - 1] - P[:, N - 2]) / (N * P[:, N - 1])

        w = 2.0 / ((N - 1) * N * np.square(P[:, N - 1]))
        x = (x[::-1] - x) / 2
        w = (w[::-1] + w) / 2
        return (x, w)