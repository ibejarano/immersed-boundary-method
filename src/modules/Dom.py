from petsc4py import PETSc

try:
    import cPickle as pickle
except ImportError:
    import pickle

class Domain2(object):
    """Domain of the problem to be solved."""

    def __init__(self, pteFile, boundFile, comm=PETSc.COMM_WORLD):
        """Constructor of the class Domain2."""
        self.onlyCornerNodes = True

        self.cfgFile = boundFile

        config = configparser.RawConfigParser()
        with open(boundFile) as fb:
            config.readfp(fb)

        # Read the binary output from the mesh generator
        # p: coordinates
        # t: connectivity
        # e: boundary conditions of the nodes numbered as in the config file

        # In the case that no mesh is defined read it from the cfg file.
        if not len(pteFile):
            pteFile = config.get('General', 'mesh')

        with open(pteFile, 'rb') as fin:
            p, t, e = pickle.load(fin)

        self.npoin = p.shape[0]
        self.nelem = t.shape[1]
        # self.nnode = None
        self.dm = None
        self.dim = 2

        # bcDict = dict()
        # bcList = list()
        self.comm = comm

        elemArray = t.transpose()
        coordArray = p

        # Only Proc 0 reads mesh file
        if not self.comm.rank:
            # Examples of boundary conditions
            # left
            # x=0
            # z=functionName // function MUST be in the file userinput.py
            # If the variable is not defined, then it's free

            if self.comm.size is not 1:
                # Proc 0 sends mesh info to others
                infoArray = np.array([self.npoin, self.nelem, self.dim])
                infoArray = self.comm.bcast(infoArray, root=0)

        else:
            # Proc != 0 receives mesh info from 0
            infoArray = None
            infoArray = comm.bcast(infoArray, root=0)

            self.npoin = infoArray[0]
            self.nelem = infoArray[1]
            self.dim = infoArray[2]

        # Create a DMPlex object
        self.dm = PETSc.DMPlex()

        # Only the one with rank=0 creates the DMPlex. The other(s) are empty.
        if self.comm.rank:
            # FIXME The method below does not work with elements with more
            # than 4 nodes.
            self.dm.createFromCellList(self.dim,
                                       np.zeros((0, self.nnode),
                                                dtype=PETSc.IntType),
                                       np.zeros((0, self.dim),
                                                dtype=float),
                                       comm=self.comm)

            # Proc != 0 receives DMPlex info from 0
            infoArray = None
            infoArray = comm.bcast(infoArray, root=0)

            self.dmNumVert = infoArray[0]
            self.dmNumEdge = infoArray[1]
            self.dmNumCell = infoArray[2]

        else:
            self.dm.createFromCellList(self.dim, elemArray, coordArray,
                                       comm=self.comm)

            vStart, vEnd = self.dm.getDepthStratum(0)
            eStart, eEnd = self.dm.getDepthStratum(1)
            cStart, cEnd = self.dm.getDepthStratum(2)

            self.dmNumVert = vEnd - vStart
            self.dmNumEdge = eEnd - eStart
            self.dmNumCell = cEnd - cStart

            if self.comm.size is not 1:
                # Proc 0 sends DMPlex info to others
                infoArray = np.array([self.dmNumVert, self.dmNumEdge,
                                      self.dmNumCell])
                infoArray = self.comm.bcast(infoArray, root=0)

            # # bcTypes = (Dirichlet)
            # for bcName in bcTypes:
            #     self.dm.createLabel(bcName)

            bcName = 'cfgfileBC'
            self.dm.createLabel(bcName)

            # for bcnum in config.sections():
            for bcnum in e.keys():
                # Take into account only the sections called 'bc-XX' that
                # appear in e{}
                if not bcnum.startswith('bc-'):
                    continue
                try:
                    bcnumInt = int(bcnum[3:])
                except:
                    continue

                # # FIXME: is nonsense to discriminate TD and Dir here
                # if (not config.has_option(bcnum, 'timedependent') or
                #         not config.getboolean(bcnum, 'timedependent')):
                #     bcType = 'Dirichlet'
                # else:
                #     bcType = 'TimeDependent'

                # Loop through the list of nodes
                for n in e[bcnum][0]:
                    # Add to DMPlex or merge in case that it is already in
                    # bitwise operator (&)
                    oldVal = self.dm.getLabelValue(bcName, n + vStart)
                    if oldVal >= 0:
                        self.dm.clearLabelValue(bcName, n + vStart, oldVal)
                        self.dm.setLabelValue(bcName, n + vStart,
                                              2**bcnumInt | oldVal)
                    else:
                        self.dm.setLabelValue(bcName, n + vStart,
                                              2**bcnumInt)

            # for bcName in bcTypes:
            if self.dm.getLabelSize(bcName):
                edges = list()
                nodeBCset = set()
                for labVal in self.dm.getLabelIdIS(bcName).getIndices():
                    nodeBCset |= set(self.dm.getStratumIS(bcName, labVal).
                                     getIndices())

                for doubleTern in itertools.combinations(nodeBCset, 2):
                    newEdge = self.dm.getJoin(doubleTern)
                    if newEdge:
                        # I have to check the BCs of both nodes and add
                        # only the ones present in both nodes to the edge
                        bcEdge = self.dm.getLabelValue(bcName,
                                                       doubleTern[0]) &\
                                 self.dm.getLabelValue(bcName,
                                                       doubleTern[1])
                        edges.append((newEdge, bcEdge))

                for edg in edges:
                    # edg is a tuple with the edge index and the BC
                    # related to it
                    self.dm.setLabelValue(bcName, edg[0], edg[1])
            else:
                self.dm.removeLabel(bcName)
