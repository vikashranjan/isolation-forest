using Printf,   StatsBase , Random, Distributions, LinearAlgebra,Base.Threads

function c_factor(n::Int)
    return 2.0*(log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))
end

function depth_unsuccessful_search(n::Int)
    return convert(Int,ceil(log(2,n)))
end

function len(X::Array)
    return size(X)[1]
end
function dim(X::Array)
    return size(X)[2]
end
mutable struct Node

    X::Array # Data at the node.
    n::Array # Normal vector used to build the hyperplane that splits the data in the node.
    p::Array # Intercept point through which the hyperplane passes.
    e::Float64 # Depth of the tree to which the node belongs.

    size::Int # Size of the dataset present at the node.
    ntype::String #Specifies if the node is external or internal. Takes two values: 'exNode', 'inNode'.

    left::Node # Left child node.
    right::Node # Right child node.
    

    Node(X, n, p, e) = new(X,n,p,e,len(X),"")

    Node(X, n, p, e, node_type,left, right) = new(X,n,p,e,len(X),node_type,left,right)

    Node(X, n, p, e,node_type) = new(X,n,p,e,len(X),node_type)

end

mutable struct iTree
    X::Array # Data at the node.
    e::Int # Depth of the tree to which the node belongs.
    l::Int # Maxium depth a tree can reach before its creation is terminated.
    exlevel::Int # Extension level used in the splitting criteria.

    size::Int # Size of the dataset.
    dim::Int # Dimension of the dataset.
    Q::Array # List of ordered integers smaller than dim.

    exnodes::Int # The number of external nodes this tree has.

    n::Array # Normal vector at the root of this tree, which is used in creating hyperplanes for splitting critera
    p::Array # Intercept point at the root of this tree through which the splitting hyperplane passes.
    
    root::Node #  At each node create a new tree.

    iTree(X,e,l,exlevel) = new(X,e,l,exlevel,len(X),dim(X),Array(range(1, stop=len(X), step=1)),0) 
end

function(t::iTree)()
    t.root=make_tree!(t,t.X,t.e,t.l)
end

function make_tree!(t::iTree,X,e,l)
    t.e = e
    if e >= l || len(X) <= 1                                              # A point is isolated in traning data, or the depth limit has been reached.
        t.exnodes += 1
        return Node(X, t.n, t.p, e,"exNode")
    else                                                                   # Building the tree continues. All these nodes are internal.
        mins =  minimum(X, dims=1)
        maxs =  maximum(X, dims=1)
        idxs = sample(1:t.dim, t.dim-t.exlevel-1,replace=false)            # Pick the indices for which the normal vector elements should be set to zero acccording to the extension level. 
        t.n = rand(Normal(0, 1),t.dim)                                     # A random normal vector picked form a uniform n-sphere. Note that in order to pick uniformly from n-sphere, we need to pick a random normal for each component of this vector.
        t.n[idxs] .= 0
        t.p = [if min < max rand(Uniform(min, max)) else min end for (min, max) in zip(mins,maxs)]   # Picking a random intercept point for the hyperplane splitting data.
        w0=X.-t.p
        w=w0 * t.n .< 0                                                    # Criteria that determines if a data point should go to the left or right child node.
        return Node(X, t.n, t.p, e,"inNode",
        make_tree!(t,X[w,:],e+1,l),
        make_tree!(t,X[.!w,:],e+1,l),
        )
    end
    return t
end


mutable struct PathFactor

    x::Array # A single data point, which is represented as a list of floats.
    itree::iTree # Given Tree
    path_list::Array{String} #Path List
    e::Float64 #  The depth of a given node in the tree.
    path::Float64 # Path Length
    PathFactor(x::Array,itree::iTree) = new(x,itree,String[],0,0)
end

function (f::PathFactor)()
    f.path=find_path!(f,f.itree.root)
end

function find_path!(f::PathFactor,T::Node)

    if T.ntype == "exNode"
        
        if T.size <= 1 
            return f.e
        else
            f.e = f.e + c_factor(T.size)
            return f.e
        end

    else
        p = T.p                                                             # Intercept for the hyperplane for splitting data at a given node.
        n = T.n                                                             # Normal vector for the hyperplane for splitting data at a given node.

        f.e += 1

        if ((transpose(f.x)-p) * n)[1] < 0
            push!(f.path_list,"L")
            return find_path!(f,T.left)
        else
            push!(f.path_list,"R")
            return find_path!(f,T.right)
        end
    end
            
end

struct iForest
    X::Array # Data used for training. It is a list of list of floats.
    nobjs::Int # Size of the dataset.
    sample::Int # Size of the sample to be used for tree creation.
    ntrees::Int # A list of tree objects.
    limit::Int # Maximum depth a tree can have.
    exlevel::Int # Exention level to be used in the creating splitting critera.
    c::Float64 # Multiplicative factor used in computing the anomaly scores.
    Trees::Array{iTree}  #Array of trees
    iForest(X::Array,nobjs::Int,sample::Int,ntrees::Int,limit::Int,exlevel::Int,c::Float64 ) = new(X,nobjs,sample,ntrees,limit,exlevel,c,iTree[])
end


iForest(X::Array,sample::Int,ntrees::Int,limit::Int,exlevel::Int) = iForest(X,len(X),sample,ntrees,limit,exlevel,c_factor(sample))

iForest(X::Array,sample::Int,ntrees::Int,limit::Int) = iForest(X,len(X),sample,ntrees,limit,0,c_factor(sample))

iForest(X::Array,sample::Int,ntrees::Int) = iForest(X,len(X),sample,ntrees,depth_unsuccessful_search(sample),0,c_factor(sample))

iForest(X::Array,sample::Int) = iForest(X,len(X),sample,100,depth_unsuccessful_search(sample),0,c_factor(sample))

function check_extension_level(forest::iForest)
    
    dim = size(forest.X)[1]
    if forest.exlevel < 0 || forest.exlevel > dim-1
        throw(ArgumentError( @printf("Extension level has to be an integer between 0 and  %f \n", dim-1)))
    end

end

function (f::iForest)()

    check_extension_level(f)

    Threads.@threads for i in 1:f.ntrees                                          # This loop builds an ensemble of iTrees (the forest).
        ix = sample(1:f.nobjs, f.sample)
        X_p = X[ix,:]
        tree=iTree(X_p, 0, f.limit, f.exlevel)
        tree()
        push!(f.Trees,tree)
    end
    return f
end

function compute_paths(f::iForest,X_in::Array)
    S = zeros(len(X_in))
    for i in  1:len(X_in)
        h_temp = 0
        for j in 1:f.ntrees
            pf=PathFactor(X_in[i,:],f.Trees[j])
            pf()
            h_temp += pf.path*1.0              # Compute path length for each point
        Eh = h_temp/f.ntrees                                             # Average of path length travelled by the point in all trees.
        S[i] = 2.0^(-Eh/f.c)                                            # Anomaly Score
        end
    end

    return S
end

function compute_paths(f::iForest)
    return compute_paths(f,f.X)
end

function compute_paths_single_tree(f::iForest,X_in::Array,tree_indx::Int)
    S = zeros(len(X_in))
    for i in  1:len(X_in)
        pf=PathFactor(X_in[i,:],f.Trees[tree_indx])
        pf()
        h_temp = pf.path*1.0              # Compute path length for each point                                   
        S[i] = h_temp                                                 # Anomaly Score
    end
    
    return S
end

function compute_paths_single_tree(f::iForest,tree_indx::Int)
    return compute_paths_single_tree(f,f.X,tree_indx)
end
