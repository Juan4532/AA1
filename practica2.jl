
# Wisconsin Diagnostic Breast Cancer (WDBC)


using DelimitedFiles
using Statistics
using Flux
using Flux.Losses


# Lectura del dataset
dataset = readdlm("P1\\wdbc.data",',');

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

#################################
# HASTA ESTE PUNTO ESTA REVISADO
#################################

function trainClassANN(topology::AbstractArray{<:Int,1},
                       dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
                       maxEpochs::Int = 1000, minLoss::Real = 0, learningRate::Real = 0.01)

    # Crea la RNA (pesos inicializados aleatoriamente)
    ann = buildClassANN(topology, size(dataset[1], 2), size(dataset[2], 2));

    # Define la función de "loss"
    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);

    # Vector con los valores de "loss" en cada ciclo de entrenamiento
    losses = [];

    # Valor de "loss" sobre la red sin entrenar
    error = loss(dataset[1]', dataset[2]');

    # Entrena la red hasta que alcance un error de entrenamiento aceptable
    while error > minLoss
        Flux.train!(loss, params(ann), [(dataset[1]', dataset[2]')], ADAM(learningRate));
        error = loss(dataset[1]', dataset[2]'); # Valor de "loss" tras el ciclo
        append!(losses, error);
    end

    return ann, losses
end

function trainClassANN(topology::AbstractArray{<:Int,1},
                       dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
                       maxEpochs::Int = 1000, minLoss::Real = 0, learningRate::Real = 0.01)

    # Convierte el vector de salidas deseadas a una matriz con una columna
    dataset[2] = reshape(dataset[2], length(dataset[2]), 1);

    # Llamada a la función anterior
    trainClassANN(topology, dataset, maxEpochs, minLoss, learningRate);
end


# Puesto que los valores son el resultado de mediciones,
# optaremos por hacer uso de la normalización de media 0

# Entradas normalizadas por media 0
inputsNorm = normalizeZeroMean(inputs)

# Salidas deseadas codificadas
targets = oneHotEncoding(targets)


topology = [1]
ann, losses = trainClassANN(topology, (inputsNorm, targets), minLoss = 0.15)

outputs = ann(inputsNorm')
accuracy(targets, outputs')
