class Mat(object):
    def __init__(self, dom):
        self.dom = dom

    def buildMatrices(self):
            """Build Matrices."""
            # FIXME: solve what happens when one node has many BC to be imposed
            self.logger.debug("readBoundaryCondition")
            self.BCdict, self.tag2BCdict, self.node2tagdict, self.BC2nodedict,\
                self.nodeROTdict = self.readBoundaryCondition()

            self.logger.debug("buildMatrices")
            WeigSrT, WeigDivSrT, WeigCurl = self.createEmptyMatrices()

            nodeBCset = set(self.node2tagdict.keys())
            # nodeROTset = set(self.nodeROTdict.keys())

            indices2one = set()  # matrix indices to be set to 1 for BC imposition
            indices2onefs = set()  # idem for FS solution
            for el in range(self.nelem):
                self.logger.debug("   Element: " + str(el))
                indices, plexPoi = self.getElemGlobNodes(el)

                # Build velocity and vorticity DoF indices
                indicesVel = [x*self.dim + d for x in indices
                            for d in range(self.dim)]
                indicesW = [x*self.dim_w + d
                            for x in indices for d in range(self.dim_w)]
                indicesSrT = [x*self.dim_s + d
                            for x in indices for d in range(self.dim_s)]

                # Compute KL equation elemental matrices
                locK, locRw, locRd = self.getElemKLEMatrices(el)

                # Compute elemental differential operator matrices
                locSrT, locDivSrT, locCurl, locWei = self.getElemKLEOperators(el)

                nodeBCintersect = nodeBCset & set(indices)
                dofFreeFSSetNS = set()  # local dof list free at FS sol
                dofSetFSNS = set()  # local dof list set at both solutions
                for node in nodeBCintersect:
                    indLoc = indices.index(node)

                    # FIXME: apply BC composition rules
                    nsNorm = set()
                    for bc in self.tag2BCdict[self.node2tagdict[node][0]]:
                        try:
                            nsNorm.add(int(self.BCdict[bc]['ns']))
                        except:
                            pass

                    if nsNorm:
                        dofSetFSNS.update([indLoc*self.dim + d
                                        for d in nsNorm])
                        dofFreeFSSetNS.update([indLoc*self.dim + d
                                            for d in (set(range(self.dim)) -
                                                        nsNorm)])
                    else:
                        for bc in self.tag2BCdict[self.node2tagdict[node][0]]:
                            for pos in range(self.dim):
                                if self.BCdict[bc]['vel'][pos] is not None:
                                    dofSetFSNS.add(indLoc*self.dim + pos)

                dofFree = list(set(range(self.elemType.nnode * self.dim))
                            - dofFreeFSSetNS - dofSetFSNS)
                dof2beSet = list(dofFreeFSSetNS | dofSetFSNS)
                dofFreeFSSetNS = list(dofFreeFSSetNS)
                dofSetFSNS = list(dofSetFSNS)

                # global counterparts of dof sets
                gldofFreeFSSetNS = [indicesVel[ii] for ii in dofFreeFSSetNS]
                gldofSetFSNS = [indicesVel[ii] for ii in dofSetFSNS]
                gldof2beSet = [indicesVel[ii] for ii in dof2beSet]
                gldofFree = [indicesVel[ii] for ii in dofFree]

                if nodeBCintersect:
                    # setValues for matrix assemble add values with addv=True
                    # setValues for BC imposition fix values in 1 with False
                    self.Krhs.setValues(
                        gldofFree, gldof2beSet,
                        -locK[np.ix_(dofFree, dof2beSet)], addv=True)

                    indices2one.update(gldof2beSet)

                    # FIXME: is the code below really necessary?
                    for indd in gldof2beSet:
                        # setting values to 0 to allocate space see below
                        # setting to 1
                        self.Krhs.setValues(indd, indd, 0, addv=True)

                    if dofFreeFSSetNS:  # bool(dof2NSfs):
                        self.Kfs.setValues(gldofFreeFSSetNS, gldofFree,
                                        locK[np.ix_(dofFreeFSSetNS, dofFree)],
                                        addv=True)

                        self.Kfs.setValues(gldofFree, gldofFreeFSSetNS,
                                        locK[np.ix_(dofFree, dofFreeFSSetNS)],
                                        addv=True)

                        self.Kfs.setValues(
                            gldofFreeFSSetNS, gldofFreeFSSetNS,
                            locK[np.ix_(dofFreeFSSetNS, dofFreeFSSetNS)],
                            addv=True)

                        # Indices where diagonal entries should be reduced by 1
                        indices2onefs.update(gldofFreeFSSetNS)

                        self.Rwfs.setValues(gldofFreeFSSetNS, indicesW,
                                            locRw[dofFreeFSSetNS, :], addv=True)

                        self.Rdfs.setValues(gldofFreeFSSetNS, indices,
                                            locRd[dofFreeFSSetNS, :], addv=True)

                    if bool(dofSetFSNS) and bool(dofFreeFSSetNS):
                        self.Krhsfs.setValues(
                            gldofFreeFSSetNS, gldofSetFSNS,
                            - locK[np.ix_(dofFreeFSSetNS, dofSetFSNS)], addv=True)

                    # if dofSetFSNS:
                        self.Krhsfs.setValues(
                            gldofFree, gldofSetFSNS,
                            - locK[np.ix_(dofFree, dofSetFSNS)], addv=True)

                        for indd in gldofSetFSNS:
                            self.Krhsfs.setValues(indd, indd, 0, addv=True)

                # Elemental matrices assembled in the global distributed matrices
                self.K.setValues(gldofFree, gldofFree,
                                locK[np.ix_(dofFree, dofFree)], addv=True)

                for indd in gldof2beSet:
                    self.K.setValues(indd, indd, 0, addv=True)

                self.Rw.setValues(gldofFree, indicesW,
                                locRw[np.ix_(dofFree, range(len(indicesW)))],
                                addv=True)

                self.Rd.setValues(gldofFree, indices,
                                locRd[np.ix_(dofFree, range(len(indices)))],
                                addv=True)

                # Differential operator matrices assembled in the global matrices
                self.SrT.setValues(indicesSrT, indicesVel, locSrT, True)
                self.DivSrT.setValues(indicesVel, indicesSrT, locDivSrT, True)
                self.Curl.setValues(indicesW, indicesVel, locCurl, True)

                WeigSrT.setValues(indicesSrT, np.repeat(locWei, self.dim_s), True)
                WeigDivSrT.setValues(indicesVel, np.repeat(locWei, self.dim), True)
                WeigCurl.setValues(indicesW, np.repeat(locWei, self.dim_w), True)

            self.K.assemble()
            self.Rw.assemble()
            self.Rd.assemble()

            self.Krhs.assemble()

            # Setting values to one can not be done in the same pass that
            # assembling matrix values because addv=False can not be mixed with
            # addv=True without an intermediate assemble
            for indd in indices2one:
                self.Krhs.setValues(indd, indd, 1, addv=False)
                self.K.setValues(indd, indd, 1, addv=False)
            self.Krhs.assemble()
            self.K.assemble()

            if self.globalNSbc:
                for indd in indices2onefs:
                    self.Kfs.setValues(indd, indd, -1, addv=True)
                self.Kfs.assemble()
                self.Rwfs.assemble()
                self.Rdfs.assemble()

                self.Krhsfs.assemble()
                for indd in (indices2one - indices2onefs):
                    self.Krhsfs.setValues(indd, indd, 1, addv=False)
                self.Krhsfs.assemble()

            # Scale all components with WeigXXX matrices
            self.SrT.assemble()
            WeigSrT.assemble()
            WeigSrT.reciprocal()
            self.SrT.diagonalScale(L=WeigSrT)

            self.DivSrT.assemble()
            WeigDivSrT.assemble()
            WeigDivSrT.reciprocal()
            self.DivSrT.diagonalScale(L=WeigDivSrT)

            self.Curl.assemble()
            WeigCurl.assemble()
            WeigCurl.reciprocal()
            self.Curl.diagonalScale(L=WeigCurl)