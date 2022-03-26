function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    VP = count(outputs.&targets)
    VN = count(.~(outputs.|targets))
    FP = count((outputs.⊻targets).&(outputs.==true)) 
    FN = count((outputs.⊻targets).&(outputs.==false)) 

    precision =  (VN + VP)/(VN + VP + FN + FP) 
    tFallo =  (FN + FP)/(VN + VP + FN + FP)
    sensibilidad =  VP/(FN+VP) 
    especifidad = VN/(FP+VN)
    vpp = VP/(VP+FP)
    vpn = VN/(VN+FN)
    f1score = 2*(vpp*sensibilidad)/(vpp+sensibilidad)
    M = [VN FP;FN VP]
    M = Int64.(M)


    if length(VN) == length(outputs)
        sensibilidad =1
        vpp = 1
    end

    if length(VP) == length(outputs)
        especifidad =1
        vpn = 1
    end

    if (vpp == 0) & (sensibilidad == 0)
        f1score =0
    end

    return replace!([precision,tFallo,sensibilidad,especifidad,vpp,vpn,f1score,M],Inf=>0,NaN=>0)
end;


confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) = confusionMatrix((outputs .>= umbral),targets)  
