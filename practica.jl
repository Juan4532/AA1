using Statistics
using DelimitedFiles


function normalizar(m)                    #Normaliza matrices por columnas
    medias = mean(inputs,dims=1)
    desviaciones = std(inputs,dims=1)
    
    norm = zeros(size(inputs))
    
    for i in 1:size(m)[2]        
        norm[:,i] = (inputs[:,i] .- medias[i]) ./ desviaciones[i]
    end
    
    return norm
end


function categoricas(x)      
    clases = unique(x)       #Almacena todas los tipos de clases que hay
    if length(clases) <= 2   #Si hay dos clases devualve un vector
        y = x .== clases[1]
    else                        #Si hay mas de dos clases devuelve una matrix
        y = zeros(size(x))
        y = convert(Array{Int64},h)
        for i in 1:length(clases)
            y[:,i] = x .== clases[i]
        end
    end
    return y
end



dataset = readdlm("wdbc.data",',');
inputs = dataset[:,3:32];
targets = dataset[:,2];

targets = categoricas(targets)

inputs = Float32.(inputs)
inputsNorm = normalizar(inputs)

###Comprobacion
print(targets)
print(size(inputsNorm))
print(mean(inputsNorm,dims=1))
print(std(inputsNorm,dims=1))
