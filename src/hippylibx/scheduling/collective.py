# Copyright (c) 2025-, Georgia Institute of Technology.
# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California, Merced.
# Copyright (c) 2019-2020, The University of Texas at Austin 
# University of California--Merced, Washington University in St. Louis.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

"""Collective scheduling utilities for MPI workloads."""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

class NullCollective:
    """
    No-overhead "Parallel" reduction utilities when a serial system of PDEs is solved on 1 process.
    """    
    def bcast(self, v, root=0):
        return v
        
    def size(self):
        return 1

    def rank(self):
        return 0

    def allReduce(self, v, op):

        if op.lower() not in ["sum", "avg"]:
            err_msg = "Unknown operation *{0}* in NullCollective.allReduce".format(op)
            raise NotImplementedError(err_msg)

        return v

    

    
class MultipleSamePartitioningPDEsCollective:
    """
    Parallel reduction utilities when several serial systems of PDEs (one per process) are solved concurrently.
    """
    def __init__(self, comm, is_serial_check=False):
        """
        :code:`comm` is :code:`mpi4py.MPI` comm
        """
        self.comm = comm
        self.is_serial_check = is_serial_check

    
    def size(self):
        return self.comm.Get_size()
    
    def rank(self):
        return self.comm.Get_rank()

    def _allReduce_array(self,v,op):
        err_msg = "Unknown operation *{0}* in MultipleSerialPDEsCollective.allReduce".format(op)
        receive = np.zeros_like(v)
        self.comm.Allreduce(v, receive, op = MPI.SUM)
        if op == "sum":
            v[:] = receive
        elif op == "avg":
            v[:] = (1./float(self.size()))*receive
        else:
            raise NotImplementedError(err_msg)            
        return v

    
    def allReduce(self, v, op):
        """
        Case handled:
        - :code:`v` is a scalar (:code:`float`, :code:`int`);
        - :code:`v` is a numpy array (NOTE: :code:`v` will be overwritten)
        - :code:`v` is a :code:`PETSc.Vec` (NOTE: :code:`v` will be overwritten)
        Operation: :code:`op = "Sum"` or `"Avg"` (case insentive).
        """
        op = op.lower()
        
        
        if isinstance(v, (float, np.floating)):
            v_array = np.array([v], dtype=np.float64)
            self._allReduce_array(v_array, op)
            return v_array[0]
                
        elif isinstance(v, (int, np.integer)):
            v_array = np.array([v], dtype=np.int64)
            self._allReduce_array(v_array, op)
            return v_array[0]
        
        elif isinstance(v, np.ndarray):
            return self._allReduce_array(v,op)
              
        elif isinstance(v, PETSc.Vec):
            if self.is_serial_check:
                assert v.getComm().Get_size() == 1
            v_array = v.array.copy()
            self._allReduce_array(v_array, op)
            v.array[:] = v_array
            v.assemblyBegin()
            v.assemblyEnd()
            return v
        elif hasattr(v,'nvec'):
            for i in range(v.nvec()):
                self.allReduce(v[i],op)
            return v
        else:
            if self.is_serial_check:
                msg = "MultipleSerialPDEsCollective.allReduce not implement for v of type {0}".format(type(v))
            else:
                msg = "MultipleSamePartitioningPDEsCollective.allReduce not implement for v of type {0}".format(type(v))
            raise NotImplementedError(msg) 

    def bcast(self, v, root = 0):
        """
        Case handled:
        - :code:`v` is a scalar (:code:`float`, :code:`int`);
        - :code:`v` is a numpy array (NOTE: :code:`v` will be overwritten)
        - :code:`v` is a :code:`PETSc.Vec` (NOTE: :code:`v` will be overwritten)
        - :code:`root` refers to the process rank within the communicator for which the data to be
        broadcasted lives.
        """
        
        if isinstance(v, (float, np.floating, int, np.integer)):
            v_array = np.array([v])
            self.comm.Bcast(v_array,root = root)
            return v_array[0]
        
        if isinstance(v, np.ndarray):
            self.comm.Bcast(v,root = root)
            return v
              
        elif isinstance(v, PETSc.Vec):
            if self.is_serial_check:
                assert v.getComm().Get_size() == 1
            v_local = v.array.copy()
            self.comm.Bcast(v_local, root=root)
            v.array[:] = v_local
            v.assemblyBegin()
            v.assemblyEnd()
            return v
        elif hasattr(v,'nvec'):
            for i in range(v.nvec()):
                self.bcast(v[i],root=root)
            return v

        else:
            if self.is_serial_check:
                msg = "MultipleSerialPDEsCollective.bcast not implement for v of type {0}".format(type(v))
            else:
                msg = "MultipleSamePartitioningPDEsCollective.bcast not implement for v of type {0}".format(type(v))
            raise NotImplementedError(msg) 

def MultipleSerialPDEsCollective(comm):
    return MultipleSamePartitioningPDEsCollective(comm, is_serial_check=True)


    
