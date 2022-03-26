using Statistics
using Flux
using Flux.Losses
using DelimitedFiles


function oneHotEncoding(feature,clases)
    if length(clases) <= 2   #Si hay dos clases devualve un vector
        y = reshape(feature .== clases[1],(length(feature),1))
    else                        #Si hay mas de dos clases devuelve una matrix
        y = zeros(length(feature),length(clases))
        y = convert(BitArray{2},y)
        for i in 1:length(clases)
            y[:,i] = feature .== clases[i]
        end
    end
    return y
end

function oneHotEncoding(feature::AbstractArray{<:Any,1})
    clases = unique(feature)
    oneHotEncoding(feature, clases)
end

function oneHotEncoding(feature::AbstractArray{Bool,1})
    clases = unique(feature)
    y = reshape(feature .== clases[1],(length(feature),1))
    return y
end




function calculateMinMaxNormalizationParameters(x::AbstractArray{<:Real,2})
    maximos = maximum(inputs,dims=1)
    minimos = minimum(inputs,dims=1)
    return (maximos,minimos)
end

function calculateZeroMeanNormalizationParameters(x::AbstractArray{<:Real,2})
    medias = mean(inputs,dims=1)
    desviaciones = std(inputs,dims=1)
    return (medias,desviaciones)
end



function normalizeMinMax!(matriz::AbstractArray{<:Real,2},
                            parametros::NTuple{2, AbstractArray{<:Real,2}})

    for i in 1:size(matriz,2)
        matriz[:,i] = (matriz[:,i] .- parametros[2][i]) ./ (parametros[1][i].-parametros[2][i])
    end
    return matriz
end

function normalizeMinMax!(matriz::AbstractArray{<:Real,2})

    parametros = calculateMinMaxNormalizationParameters(matriz)

    normalizeMinMax!(matriz,parametros)
end

function normalizeMinMax(matriz::AbstractArray{<:Real,2},
                            parametros::NTuple{2, AbstractArray{<:Real,2}})

    matriz1 = copy(matriz)

    for i in 1:size(matriz1,2)
        matriz1[:,i] = (matriz1[:,i] .- parametros[2][i]) ./ (parametros[1][i].-parametros[2][i])
    end
    return matriz1
end

function normalizeMinMax(matriz::AbstractArray{<:Real,2})

    parametros = calculateMinMaxNormalizationParameters(matriz1)

    normalizeMinMax(matriz,parametros)
end


function normalizeZeroMean!(matriz::AbstractArray{<:Real,2},
                            parametros::NTuple{2, AbstractArray{<:Real,2}})

    for i in 1:size(matriz,2)
        matriz[:,i] = (matriz[:,i] .- parametros[1][i]) ./ parametros[2][i]
    end
    return matriz
end

function normalizeZeroMean!(matriz::AbstractArray{<:Real,2})

    parametros = calculateZeroMeanNormalizationParameters(matriz)

    normalizeZeroMean!(matriz,parametros)
end

function normalizeZeroMean(matriz::AbstractArray{<:Real,2},
                            parametros::NTuple{2, AbstractArray{<:Real,2}})

    matriz1 = copy(matriz)

    for i in 1:size(matriz1,2)
        matriz1[:,i] = (matriz1[:,i] .- parametros[1][i]) ./ parametros[2][i]
    end
    return matriz1
end

function normalizeZeroMean(matriz::AbstractArray{<:Real,2})

    parametros = calculateZeroMeanNormalizationParameters(matriz)

    normalizeZeroMean(matriz,parametros)
end


function classifyOutputs(outputs::AbstractArray{<:Real,2},umbral=0.5)
    if size(outputs,2) == 1
        outputs = outputs .>= umbral

    else
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
    end
    return outputs
end

function accuracy(targets::AbstractArray{Bool,1},outputs::AbstractArray{Bool,1})
    classComparison = targets .== outputs
    correctClassifications = all(classComparison, dims=2)
    prec = mean(correctClassifications)
    return prec
end

function accuracy(targets::AbstractArray{Bool,2},outputs::AbstractArray{Bool,2})
    if typeof(targets) .== typeof(outputs) == BitArray{1}
        accuracy(targets,outputs)

    else
        classComparison = targets .== outputs
        correctClassifications = all(classComparison, dims=2)
        prec = mean(correctClassifications)
        return prec
    end
end

function accuracy(targets::AbstractArray{Bool,1},outputs::AbstractArray{<:Real,1}, umbral=0.5)
    outputs = classifyOutputs(outputs,umbral)
    accuracy(targets,outputs)
end

function accuracy(targets::AbstractArray{Bool,2},outputs::AbstractArray{<:Real,2})
    if typeof(targets) .== typeof(outputs) == BitArray{1}
        accuracy(targets,outputs)

    else
        outputs = classifyOutputs(outputs)
        accuracy(targets,outputs)
    end
end

function rrnnaa(topology::AbstractArray{<:Int,1},numInput,numOutput)
    ann = Chain();
    numInputsLayer = numInput;
    for numOutputsLayer = topology
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, σ) );
        numInputsLayer = numOutputsLayer;
    end;

    if numOutput == 1
        ann = Chain(ann..., Dense(numInputsLayer, numOutput ,σ))
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutput ,identity))
        ann = Chain(ann..., softmax)
    end

    return ann
end

function clasificacion(topology::AbstractArray{<:Int,1},
            dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
            maxEpochs::Int=1000, minLoss::Real =0.3, learningRate::Real = 0.01)

    ann = rrnnaa(topology,size(dataset[1],2),size(dataset[2],2))

    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);

    loses= []

    for i in 1:maxEpochs
        Flux.train!(loss, params(ann), [(dataset[1]', dataset[2]')], ADAM(learningRate));
        error = loss(dataset[1]',dataset[2]')
        append!(loses,error)
    end
    return (ann,loses)
end


function clasificacion(topology::AbstractArray{<:Int,1},
            dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}},
            maxEpochs::Int=1000, minLoss::Real =0.15, learningRate::Real = 0.01)

    dataset[2] = reshape(dataset[2],(length(dataset[2]),1))

    clasificacion(topology,dataset,maxEpochs,minLoss,learningRate)

end


#Cargar los datos
datos = readdlm("/home/juan/uni/aa1/pra/wdbc.data",',');
inputs = datos[:,3:32];
targets = datos[:,2];

#Transformas las variables categoricas a boleanos
targets = oneHotEncoding(targets)
inputs = Float32.(inputs)
inputs = normalizeZeroMean(inputs)
dataset = (inputs,targets)

topology = [6,3]
ann,error = clasificacion(topology,dataset)
outputs = classifyOutputs(ann(inputs')')
accuracy(targets,outputs)
