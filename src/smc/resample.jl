### particle filter: resample populations ###
# GIVEN CUMULATIVE WEIGHTS

## basic multinomial resampler (given cumulative weights, i.e. inverse CDF)
function rsp_multinomial(m_pop::Array{Int64,2}, old_p::Array{Int64,2}, cw::Array{Float64,1})
    # update old p
    # old_p .= m_pop
    # choose new p
    for p in eachindex(weights)
        new_p = length(weights)
        chs_p = rand() * weights[end]
        for p2 in 1:(length(weights) - 1)
            if chs_p < weights[p2]
                new_p = p2
                break
            end
        end
        m_pop[p,:] .= old_p[new_p,:]
    end
end

## systematic (samples single seed u(0,1/N])
# Carpenter (1999)
function rsp_systematic(m_pop::Array{Int64,2}, old_p::Array{Int64,2}, cw::Array{Float64,1})
    # output = Array{Int64,1}(undef, length(cw))
    u = Array{Float64,1}(undef, length(cw))
    u[1] = rand() / length(cw) # sample ~ U(0,1/N]
    for i in 2:length(cw)
        u[i] = u[1] + ((i - 1) / length(cw))
    end
    u .*= cw[end]
    # set output = new index
    j = 1
    for i in 1:size(m_pop, 1)
        while u[i] > cw[j]
            j = j + 1
        end
        m_pop[i,:] .= old_p[j,:]
        # output[i] = j
    end
    # return output
end

## stratified (i.e. jittered)
# Kitagawa (1996)
function rsp_stratified(m_pop::Array{Int64,2}, old_p::Array{Int64,2}, cw::Array{Float64,1})
    # output = Array{Int64,1}(undef, length(cw))
    u = rand(length(cw)) / length(cw)
    for i in eachindex(u)
        u[i] += ((i - 1) / length(cw))
    end
    u .*= cw[end]
    # set output = new index
    j = 1
    for i in eachindex(output)
        while u[i] > cw[j]
            j = j + 1
        end
        # output[i] = j
        m_pop[i,:] .= old_p[j,:]
    end
    # return output
end

### OTHERS

## basic multinomial resampler (inverse CDF method)
function rs_multinomial(w::Array{Float64,1}, n::Int64 = length(w))
    cumsum!(w, w)
    output = Array{Int64,1}(undef, n)
    # choose new p
    for p in eachindex(output)
        output[p] = length(w)
        chs_p = rand() * w[end]
        for p2 in 1:(length(w) - 1)
            if chs_p < w[p2]
                output[p] = p2
                break   # next p
            end
        end
        # m_pop[p,:] .= old_p[new_p,:]
    end
    return output
end

## residual multinomial
# add option for normalising weights? **
# function rs_residual_mn(w::Array{Float64,1})
#     w ./= sum(w)    # normalise
#     nt = Int64.(floor.(length(w) .* w)) # N~
#     output = zeros(Int64, length(w))
#     i = 1
#     for j in eachindex(nt)
#         for k in 1:nt[j]
#             output[i] = nt[j]
#         end
#     end
#     wb = w .- (nt .* (1 / length(w)))
#     cumsum!(wb, wb)
#     rs = rs_multinomial(wb, length(w) - sum(nt))
#     output[(sum(nt)+1):end] .= rs
#     return output
# end
## NB. W.I.P.

## systematic (samples single seed u(0,1/N])
# Carpenter (1999)
function rs_systematic(w::Array{Float64,1})
    cw = cumsum(w)
    output = Array{Int64,1}(undef, length(w))
    u = Array{Float64,1}(undef, length(w))
    u[1] = rand() / length(w) # sample ~ U(0,1/N]
    for i in 2:length(cw)
        u[i] = u[1] + ((i - 1) / length(w))
    end
    u .*= cw[end]
    # set output = new index
    j = 1
    for i in eachindex(output)
        while u[i] > cw[j]
            j += 1
        end
        output[i] = j
    end
    return output
end

## stratified (i.e. jittered)
# Kitagawa (1996)
function rs_stratified(w::Array{Float64,1})
    cumsum!(w, w)
    output = Array{Int64,1}(undef, length(w))
    u = rand(length(w)) / length(w)
    for i in eachindex(u)
        u[i] += ((i - 1) / length(w))
    end
    u .*= w[end]
    # set output = new index
    j = 1
    for i in eachindex(output)
        while u[i] > w[j]
            j = j + 1
        end
        output[i] = j
    end
    return output
end
