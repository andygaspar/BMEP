using JuMP
using ArgParse
using Gurobi

using Graphs
using PhyloNetworks



#basic helper functions
function read_inst(filename::String)
    dists = Float64[]
    n = -1
    open(filename) do file
        n = parse(Int64,readline(file))
        for i in 1:n, j in 1:n
            push!(dists, parse(Float64,readline(file)))
        end
    end
    return reshape(dists, (n,n))
end

function write_sol(sol::Matrix{T}, filename::String) where T <: Real
    n = size(sol)[1]
    
    open(filename, "w") do file
        for i in 1:n
            for j in 1:n
                print(file, "$(sol[i,j]) ")
            end
            println(file, "")
        end
    end
end
                
function evaluate_obj(dists::Matrix{T}, inst::Matrix{Float64}) where T <: Real
    return sum(2.0^(-dists[i,j])*inst[i,j] for j in 1:size(inst)[1], i in 1:size(inst)[1])
end        

 
#convertion between τ-matrix and graphs
function phylo_to_graph(net)
    g = SimpleGraph()
    add_vertices!(g, length(net.node))
    for e in net.edge
        add_edge!(g, e.node[1].number, e.node[2].number)
    end
    return g
end

function get_taxa_dist(g::SimpleGraph, internalidx=nothing)
    n = Int(nv(g)/2+1)
    dists = zeros(Int64,n,n)

    taxa2idx = Dict{Int64, Int64}()
    taxa = internalidx
    if internalidx === nothing
        taxa = collect(1:n)
    end

    for (idx, t) in enumerate(taxa)
        push!(taxa2idx, t => idx)
    end
    for i in taxa
        res = dijkstra_shortest_paths(g, i)
        for j in taxa
            dists[taxa2idx[i], taxa2idx[j]] = Int(round(res.dists[j]))
        end
    end
    return dists
end

function tree_from_dists(sol::Matrix{T}) where T <: Real
    n = size(sol)[1]
    copysol = Matrix{Float64}(undef, n,n)
    for j in 1:n
        for i in 1:n
            if i != j
                copysol[i,j] = sol[i,j]
            else
                copysol[i,j] = 0
            end
            
        end
    end
        
    net = PhyloNetworks.nj!(copysol, [string(i) for i in 1:n])
    return phylo_to_graph(net)
end

function dists_from_sol(sol::Matrix{Float64})
    return get_taxa_dist(tree_from_dists(sol))
end


#takes x[i,j,l] and computes τ[i,j]
function vars2dists(x::Array{JuMP.VariableRef})
    n = size(x)[1]
    sol = zeros(Float64, n, n)
    for i in 1:n, j in i+1:n
        sol[i,j] = sol[j,i] = sum(l*value.(x[i,j,l]) for l in 1:n)
    end
    return sol
end


function build_model(inst::Matrix{Float64}; kraft=true, manifold=true, triangularinq=true, relax=false)
    n = size(inst)[1]
    model = JuMP.Model(Gurobi.Optimizer)
    @variable(model,  0 <= x[1:n, 1:n, 1:n] <= 1, integer= !relax)
    

    #symmetry and zero diag
    for i in 1:n, j in 1:n, l in 1:n
        if i < j
            @constraint(model, x[i,j,l] == x[j,i,l])
        end
        if i == j || l == 1 || l == n
            @constraint(model, x[i,j,l] == 0)
 
        end
    end
    
    
    #var assignement
    for i in 1:n, j in 1:n
        if i != j
            @constraint(model, sum(x[i,j,l] for l in 1:n) == 1)
        end
    end
    
    
    if kraft
        for i in 1:n
            @constraint(model, sum((2.0^-l)*x[i,j,l] for j in 1:n, l in 1:n) == 0.5)
        end
    end
    
    if manifold
        @constraint(model, sum(l*(2.0^-l)*x[i,j,l] for i in 1:n, j in 1:n, l in 1:n) == 2*n-3)
    end
    

    if triangularinq
        for i in 1:n, j in 1:n, k in 1:n
            if length(Set((i,j,k))) < 3
                continue
            end
            @constraint(model, sum(l*x[i,j,l] for l in 1:n) + sum(l*x[j,k,l] for l in 1:n) >= 2+sum(l*x[i,k,l] for l in 1:n))
        end
    end
        
    @objective(model, Min, sum(inst[i,j]*(2.0^(n-l))*x[i,j,l] for i in 1:n, j in 1:n, l in 1:n))
    
    #MOI.set(model, MOI.AbsoluteGapTolerance(), eps())
    return (model, x)
end


function solve_model(inst::Matrix{Float64}, model::JuMP.Model, x::Array{JuMP.VariableRef};
    separate_triangular_inq=false, timelimit::Int=-1)
    n = size(inst)[1]
    heu_cnt = 1
    
    if timelimit > 0
        set_time_limit_sec(model, timelimit)
    end
    
    invalid_sol = true
    best_sol = zeros(Int, n, n)
    best_sol_val = -1
    
    function nj_heu_cb(cb_data)
        x_vals = callback_value.(cb_data, x)
        frac_dists = [sum(l*x_vals[i,j,l] for l in 1:n) for i in 1:n, j in 1:n]
        #@show frac_dists
        dists = dists_from_sol(frac_dists)
        #@show dists
        
        cur_val = evaluate_obj(dists, inst)
        if  invalid_sol || cur_val < best_sol_val-1e-8
            invalid_sol = false
            best_sol = dists
            best_sol_val = cur_val
        end
        if mod1(heu_cnt, 1000) == 1
            println("heu called")
        end
        
        heu_cnt += 1
        
        var_assign = [round(dists[i,j] == l) ? 1 : 0 for i in 1:n for j in 1:n for l in 1:n]

        
        vars = [x[i,j,l] for i in 1:n for j in 1:n for l in 1:n]
    
        
        status = MOI.submit(
            model, MOI.HeuristicSolution(cb_data), vars, var_assign
        )    
        
        #@show status
    end
    
    
    num_triangular_cuts = 0
    function separate_triangular_inq_func(x_vals::Array{Float64,3})
        
        TRI_EPS = 1e-1
        
        cuts_threads = [Vector{Tuple{Float64, Any}}() for _ in 1:Threads.nthreads()]
        Threads.@threads for j in 1:n
            for i in 1:n-1
                if i == j
                    continue
                end
                for k in i+1:n
                    if k == j
                        continue
                    end
                
                
                    lhs = sum(l*x_vals[i,j,l] for l in 1:n) + sum(l*x_vals[j,k,l] for l in 1:n) - sum(l*x_vals[i,k,l] for l in 1:n)
                    if lhs < 2-TRI_EPS
                        con = @build_constraint(sum(l*x[i,j,l] for l in 1:n) + sum(l*x[j,k,l] for l in 1:n) >= 2+sum(l*x[i,k,l] for l in 1:n))
                        push!(cuts_threads[Threads.threadid()], (lhs, con))
                        num_triangular_cuts  += 1

                    end
                end
            end
        end
        
        
        
        return [cut for (value, cut) in sort(reduce(vcat, cuts_threads), lt=(x,y)->x[1] > y[1])]
    end
            
    
    
    function user_cut_cb(cb_data)
        x_vals = callback_value.(cb_data, x)
        
        if separate_triangular_inq
            for cut in separate_triangular_inq_func(x_vals)
                MOI.submit(model, MOI.UserCut(cb_data), cut)  
            end
        end

    end
    
        
    function lazy_cut_cb(cb_data)
        x_vals = callback_value.(cb_data, x)
        if separate_triangular_inq
            for cut in separate_triangular_inq_func(x_vals)
                MOI.submit(model, MOI.LazyConstraint(cb_data), cut)  
            end
        end
    end
    
    MOI.set(model, MOI.HeuristicCallback(), nj_heu_cb)
    
    if separate_triangular_inq
        MOI.set(model, MOI.UserCutCallback(), user_cut_cb)
    end
    
    if separate_triangular_inq
        MOI.set(model, MOI.LazyConstraintCallback(), lazy_cut_cb)
    end
    JuMP.optimize!(model)
    

    @show num_triangular_cuts
    return best_sol
end

function solve_inst(inst::Matrix{Float64};
    relax::Bool=false,
    triangular_inq::Bool=false,
    timelimit::Int=-1)
model, x = build_model(inst; relax=relax, triangularinq=triangular_inq, )
return solve_model(inst, model, x, separate_triangular_inq=!triangular_inq, timelimit=timelimit)
end    


# argument parsing
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--infile"
            help = "inpurt file: number of taxa n followed by n-by-n entries"
            arg_type = String
            required = true
        "--outfile"
            help = "outputfile containg n-by-n values describing the path-lengths. If not provide the result is printed."
            arg_type = String
            default = ""
        "--timelimit"
            help = "Time limit for the solver in seconds. If not provided or negative, no time limit is imposed."
            arg_type = Int
            default = -1
    end     

    #parsed_args = Dict(k => v for (k,v) in parse_args(s))
    #print(parsed_args)
    return parse_args(s)

end

function main()
    parsed_args = parse_commandline()
    infile = parsed_args["infile"]
    outfile = parsed_args["outfile"]
    timelimit = parsed_args["timelimit"]

    inst = read_inst(infile)
    sol = solve_inst(inst, relax=false, triangular_inq=false, timelimit=timelimit)

    @show evaluate_obj(sol, inst)
    if length(outfile) > 0
        write_sol(Int.(sol), outfile)
    else
        print(sol)
    end
end

main()





