function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    @assert(size(outputs,2)==size(targets,2))
    @assert(size(outputs,2)!=2)

    if size(outputs,2) == 1
        confusionMatrix(outputs,targets)
    end

    m = zeros(5,size(outputs,2))

    for i in 1:size(outputs,2)
        for iter in 3:7
            m[iter:i] = confusionMatrix(outputs[:,i],targets[:,i])[iter]
        end
    end



end;

confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true) = 0

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # CODIGO DEL ALUMNO
end;
