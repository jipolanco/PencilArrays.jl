# # Tutorial

# This tutorial walks through the creation of a

# ## Getting started

using MPI

MPI.Initialized() || # hide
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

## TODO add image!

rank

# ## Creating distributed arrays

# ## Working with arrays

# ### Accessing and modifying data

# ### Local and global indexing

# ### Broadcasting

# ## Data transpositions

# ## Advanced topics
#
# ### Dimension permutations
#
# One may want data to be contiguous in memory along one of the non-decomposed
# dimensions.
# For instance, consider the $y$-pencil configuration in the above figure.
# [TODO]
#
# An important motivation for this feature is [distributed
# FFTs](https://github.com/jipolanco/PencilFFTs.jl). FFTs are global operations
# requiring all data points along the transformed dimensions.
# Therefore, for a given pencil configuration, one would like to perform FFTs
# along the non-decomposed dimensions.
# Evidently, FFTs are faster if they are performed over contiguous data, which
# is why it is preferrable that the non-decomposed dimensions are contiguous in
# memory.
# Note that these considerations are not limited to FFTs, but may apply to
# various other problems such as the solution of linear systems.
#
# There are different possible approaches for dealing with this issue.
# One of them, would be
#
