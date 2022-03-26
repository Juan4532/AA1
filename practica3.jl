
# Wisconsin Diagnostic Breast Cancer (WDBC)


using DelimitedFiles
using Statistics
using Flux
using Flux.Losses 
using Plots
using Random


# Lectura del dataset
dataset = readdlm("wdbc.data",',');

# Elimina el atributo "ID", puesto que no es relevante
dataset = dataset[:, 2:32]

inputs = Float32.(dataset[:, 2:31]) # Matriz de entradas
targets = dataset[:, 1] # Matriz de salidas deseadas

# En nuestro problema se dispone de una única variable
# categórica, correspondiente a la salida deseada, que codificaremos
# como 0 (benigno) y 1 (maligno), usando la técnica One-Hot Encoding

function oneHotEncoding(feature::AbstractArray{<:Any, 1}, classes::AbstractArray{<:Any, 1})
    # Únicamente 2 categorías
    if length(classes) == 2
        # Crea el vector de booleanos
        oneHot = (feature .== classes[1]);
        # Transforma el vector a una matriz bidimensional de una columna
        oneHot = reshape(oneHot, (length(oneHot), 1));

    elseif length(classes) > 2 # Más de 2 categorías
        # Crea la matriz de booleanos
        oneHot = zeros(Bool, size(feature, 1), length(classes));
        for i in 1:length(classes) # Recorre las clases
            oneHot[:, i] = (feature .== classes[i]);
        end
    end

    return oneHot;
end

# Caso en el que no se pase el atributo "classes"
oneHotEncoding(feature::AbstractArray{<:Any, 1}) = oneHotEncoding(feature::AbstractArray{<:Any, 1}, unique(feature));

# Caso en el que se pase directamente un vector de booleanos
oneHotEncoding(feature::AbstractArray{Bool, 1}) = reshape(feature, (length(feature), 1));


# Calcula los parámetros de la normalización max-min
function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real, 2}, dataInRows = true)
    ( minimum(dataset, dims=(dataInRows ? 1 : 2)), maximum(dataset, dims=(dataInRows ? 1 : 2)) );
end

calculateMinMaxNormalizationParameters(inputs)

# Calcula los parámetros de la normalización de media 0
function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real, 2}, dataInRows = true)
    ( mean(dataset, dims=(dataInRows ? 1 : 2)), std(dataset, dims=(dataInRows ? 1 : 2)) );
end

calculateZeroMeanNormalizationParameters(inputs)


# Normaliza el "dataset" con min-max
function normalizeMinMax!(dataset::AbstractArray{<:Real, 2}, normalizationParameters::NTuple{2, AbstractArray{<:Real, 2}})
    for i in 1:size(dataset, 2)
        dataset[:, i] = (dataset[:, i] .- normalizationParameters[1][i]) ./ (normalizationParameters[2][i].-normalizationParameters[1][i]);
    end

    return dataset;
end

normalizeMinMax!(dataset::AbstractArray{<:Real, 2}) = normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset));

function normalizeMinMax(dataset::AbstractArray{<:Real, 2}, normalizationParameters::NTuple{2, AbstractArray{<:Real, 2}})
    datasetCopy = copy(dataset);
    normalizeMinMax!(datasetCopy);

    return datasetCopy;
end

normalizeMinMax(dataset::AbstractArray{<:Real, 2}) = normalizeMinMax(dataset, calculateMinMaxNormalizationParameters(dataset));


# Normaliza el "dataset" con mean-std
function normalizeZeroMean!(dataset::AbstractArray{<:Real, 2}, normalizationParameters::NTuple{2, AbstractArray{<:Real, 2}})
    for i in 1:size(dataset, 2)
        dataset[:,i] = (dataset[:,i] .- normalizationParameters[1][i]) ./ normalizationParameters[2][i]
    end

    return dataset;
end

normalizeZeroMean!(dataset::AbstractArray{<:Real, 2}) = normalizeZeroMean!(dataset, calculateZeroMeanNormalizationParameters(dataset));

function normalizeZeroMean(dataset::AbstractArray{<:Real, 2}, normalizationParameters::NTuple{2, AbstractArray{<:Real, 2}})
    datasetCopy = copy(dataset);
    normalizeZeroMean!(datasetCopy);

    return datasetCopy
end

normalizeZeroMean(dataset::AbstractArray{<:Real, 2}) = normalizeZeroMean(dataset, calculateZeroMeanNormalizationParameters(dataset));


# Clasifica las salidas de la RNA
function classifyOutputs(outputs::AbstractArray{<:Real,2}, umbral = 0.5)
    if size(outputs, 2) == 1
        outputs = (outputs .>= umbral);
    else
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2);
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
    end

    return outputs
end


# Calcula la precisión de un problema con únicamente 2 clases
function accuracy(targets::AbstractArray{Bool,1}, outputs::AbstractArray{Bool,1})
    classComparison = targets .== outputs;
    accuracy = mean(classComparison);

    return accuracy # Precisión con únicamente 2 clases
end

function accuracy(targets::AbstractArray{Bool,2}, outputs::AbstractArray{Bool,2})
    if size(targets, 2) == 1 # Llamada a la función anterior
        accuracy(targets[:, 1], outputs[:, 1]);
    elseif size(targets, 2) > 2 # Número de columnas mayor que 2 (más de 2 clases)
        classComparison = targets .== outputs;
        correctClassifications = all(classComparison, dims = 2);
        accuracy = mean(correctClassifications);
    end

    return accuracy
end

# Caso en que "outputs" no tenga valores de pertenencia a 2 clases
function accuracy(targets::AbstractArray{Bool,1}, outputs::AbstractArray{<:Real,1}, umbral = 0.5)
    outputs = outputs .>= umbral; # Le pasa el umbral
    accuracy(targets, outputs);
end

# Caso en que "outputs" no tenga valores de pertenencia a N clases
function accuracy(targets::AbstractArray{Bool,2}, outputs::AbstractArray{<:Real,2})
    if size(targets, 2) == 1 # Una única columna
        accuracy(targets[:, 1], outputs[:, 1]);
    elseif size(targets, 2) > 2 # Más de 2 columnas
        outputs = classifyOutputs(outputs);
        accuracy(targets, outputs);
    end
end


# Crea una RR.NN.AA con la topología especificada
function buildClassANN(topology::AbstractArray{<:Int, 1}, numInputs::Int64, numOutputs::Int64)
    ann = Chain(); # Crea una RNA vacía
    numInputsLayer = numInputs;
    for numOutputsLayer = topology # Añade cada capa a la red
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, σ) );
        numInputsLayer = numOutputsLayer;
    end

    # Capa de salida en función del número de clases
    if numOutputs == 1 # Una única neurona de salida (2 clases)
        # Función de transferencia sigmoidal
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, σ));
    else # Añade función softmax
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
        ann = Chain(ann..., softmax);
    end

    return ann # Devuelve la red creada
end





function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{Float32,2}(undef,0,0), falses(0,0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{Float32,2}(undef,0,0), falses(0,0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, size(topology, 1)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20, showText::Bool=false)


    # Crea la RNA (pesos inicializados aleatoriamente)
    ann = buildClassANN(topology, size(trainingDataset[1], 2), size(trainingDataset[2], 2));

    # Define la función de "loss"
    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);

    # Vector con los valores de "loss" en cada ciclo de entrenamiento
    trainingLosses = [];
    validationLosses = [];
    testLosses = [];

    # Valor de "loss" sobre la red sin entrenar
    trainingError = loss(trainingDataset[1]', trainingDataset[2]');

    if size(validationDataset[1]) == (0,0)
        # Entrena la red hasta que alcance un error de entrenamiento aceptable
        while trainingError > minLoss
            Flux.train!(loss, params(ann), [(trainingDataset[1]', trainingDataset[2]')], ADAM(learningRate));
            trainingError = loss(trainingDataset[1]', trainingDataset[2]'); # Valor de "loss" tras el ciclo
            push!(trainingLosses, trainingError);

            testError = loss(testDataset[1]',testDataset[2]');
            push!(testLosses,testError);
        end

        return ann, trainingLosses, testLosses
    
    else
        copyList = []
        while true
            Flux.train!(loss, params(ann), [(trainingDataset[1]', trainingDataset[2]')], ADAM(learningRate));
            trainingError = loss(trainingDataset[1]', trainingDataset[2]'); # Valor de "loss" tras el ciclo
            push!(trainingLosses, trainingError);

            validationError = loss(validationDataset[1]',validationDataset[2]');
            push!(validationLosses,validationError);

            testError = loss(testDataset[1]',testDataset[2]');
            push!(testLosses,testError);
            
            if validationError > minimum(validationLosses)
                copia = deepcopy(ann)
                push!(copyList,copia)
                if length(copyList) >= maxEpochsVal
                    return copyList[1] , trainingLosses, testLosses, validationLosses
                end
            else 
                copyList = []
            end
        end
    end
end;


function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{Float32,2}(undef,0,0), falses(0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{Float32,2}(undef,0,0), falses(0)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20, showText::Bool=false)


    trainingDataset[2] = reshape(trainingDataset[2],(length(trainingDataset[2]),1))
    validationDataset[2] = reshape(validationDataset[2],(length(validationDataset[2]),1))
    testDataset[2] = reshape(testDataset[2],(length(testDataset[2]),1))

    trainingDataset(topology,trainingDataset,validationDataset,testDataset,maxEpochs,minLoss,learningRate,maxEpochsVal,showText)


end;

function holdOut(N::Int,P::Float64)
    ind = randperm(N)
    n = Int.(round(N*(1-P)))
    training = ind[1:n]
    test = ind[n+1:N]
    tupla = (training, test)
    return tupla
end

function holdOut(N::Int,Pval::Float64,Ptest::Float64)
    a = holdOut(N,Pval)
    val = a[2]
    n = Int.(round(N*Ptest))
    p = n/length(a[1])
    et = holdOut(length(a[1]),p)
    entrenamiento = a[1][et[1]]
    test = a[1][et[2]]
    tupla = (entrenamiento, val, test)
    return tupla
end


# Puesto que los valores son el resultado de mediciones,
# optaremos por hacer uso de la normalización de media 0

# Entradas normalizadas por media 0
inputsNorm = normalizeZeroMean(inputs)

# Salidas deseadas codificadas
targets = oneHotEncoding(targets)


topology = [1]


a, b, c = holdOut(size(inputsNorm,1),0.2,0.1)

ann, trainingLosses,testLosses, validationLosses = trainClassANN(topology, (inputsNorm[a,:],targets[a,:]),testDataset=(inputsNorm[b,:],targets[b,:]),validationDataset=(inputsNorm[c,:],targets[c,:]),minLoss=0.2)


g = plot()
plot!(g,1:length(trainingLosses),trainingLosses,label="Error de entrenamiento")
plot!(g,1:length(testLosses),testLosses,label="Error de tes")
plot!(g,1:length(validationLosses),validationLosses,label="Error de entrenamiento")

display(g)



