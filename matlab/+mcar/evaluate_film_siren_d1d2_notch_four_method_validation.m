function evaluate_film_siren_d1d2_notch_four_method_validation( ...
        subjectLimit, outputName, filmPredictionName, mcarPredictionName, ...
        ranfPredictionName, fspPredictionName)
%EVALUATE_FILM_SIREN_D1D2_NOTCH_FOUR_METHOD_VALIDATION Strict 44-subject
% validation comparison of four methods only: the new FiLM-SIREN
% D1/D2+notch E130 1/3 ensemble, MCAR v3.5.1, RANF, and FSP-AE.
%
% This is the dedicated four-method entry required by the C4 common-score
% correction; it never constructs or reads v1/v2 predictions and never opens
% the locked test split. All outputs record test_subject_count_read=0.
%
% Prediction formats:
%   film/mcar: HDF5 '/predicted_residual_db' [2, 793, 463] residual dB.
%   fsp      : HDF5 '/predicted_hrir' [793, 2, 256] and
%              '/predicted_magnitude_db' [793, 2, 512].
%   ranf     : SOFA file with 793 x 2 x 256 HRIR and 793 source positions.

if nargin < 1 || isempty(subjectLimit)
    subjectLimit = inf;
end
if nargin < 2 || isempty(outputName)
    outputName = 'sonicom_film_siren_gl_final_d1d2_notch_...';
    outputName = [ ...
        'sonicom_film_siren_gl_final_d1d2_notch_' ...
        'vs_ranf_fsp_v351_validation'];
end
if nargin < 3 || isempty(filmPredictionName)
    filmPredictionName = [ ...
        'sonicom_film_siren_gl_final_d1d2_notch_' ...
        'e130_ensemble_validation'];
end
if nargin < 4 || isempty(mcarPredictionName)
    mcarPredictionName = 'sonicom_q26_validation_v351_previous30_b70';
end
if nargin < 5 || isempty(ranfPredictionName)
    ranfPredictionName = 'sonicom_ranf_q26_validation_frozen';
end
if nargin < 6 || isempty(fspPredictionName)
    fspPredictionName = 'sonicom_fsp_ae_q26_formal_validation';
end
validateattributes(subjectLimit, {'numeric'}, {'scalar', 'positive'});
validateattributes(outputName, {'char', 'string'}, {'scalartext'});
validateattributes(filmPredictionName, {'char', 'string'}, {'scalartext'});
validateattributes(mcarPredictionName, {'char', 'string'}, {'scalartext'});
validateattributes(ranfPredictionName, {'char', 'string'}, {'scalartext'});
validateattributes(fspPredictionName, {'char', 'string'}, {'scalartext'});
outputName = char(outputName);
filmPredictionName = char(filmPredictionName);
mcarPredictionName = char(mcarPredictionName);
ranfPredictionName = char(ranfPredictionName);
fspPredictionName = char(fspPredictionName);

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));
supdeqDir = fullfile(projectRoot, 'external', 'SUpDEq');
datasetRoot = fullfile(projectRoot, 'data', 'processed', ...
    'sonicom_residual_q26_v1');
sofaRoot = fullfile(projectRoot, 'data', 'HRTF', ...
    'sonicom_measured_ffcmp_minphase_44k1', 'subjects');
splitFile = fullfile(projectRoot, 'configs', 'data', ...
    'sonicom_subject_split_v1.csv');
filmRoot = fullfile(projectRoot, 'artifacts', 'reconstruction', ...
    filmPredictionName);
mcarRoot = fullfile(projectRoot, 'artifacts', 'reconstruction', ...
    mcarPredictionName);
ranfRoot = fullfile(projectRoot, 'artifacts', 'reconstruction', ...
    ranfPredictionName);
fspRoot = fullfile(projectRoot, 'artifacts', 'reconstruction', ...
    fspPredictionName);
outputRoot = fullfile(projectRoot, 'results', outputName);
figuresRoot = fullfile(outputRoot, 'figures');

assert(isfolder(supdeqDir), 'SUpDEq directory not found: %s', supdeqDir);
assert(isfolder(datasetRoot), 'Dataset root not found: %s', datasetRoot);
assert(isfolder(sofaRoot), 'SOFA root not found: %s', sofaRoot);
assert(isfile(splitFile), 'Split file not found: %s', splitFile);
assert(isfolder(filmRoot), 'FiLM-SIREN prediction root not found: %s', filmRoot);
assert(isfolder(mcarRoot), 'MCAR prediction root not found: %s', mcarRoot);
assert(isfolder(ranfRoot), 'RANF prediction root not found: %s', ranfRoot);
assert(isfolder(fspRoot), 'FSP-AE prediction root not found: %s', fspRoot);
if ~isfolder(figuresRoot)
    mkdir(figuresRoot);
end

originalDir = pwd;
restoreDirectory = onCleanup(@() cd(originalDir));
cd(supdeqDir);
supdeq_start;
cd(projectRoot);

splitTable = readtable(splitFile, 'TextType', 'string');
splitRows = splitTable(splitTable.split == "val", :);
assert(height(splitRows) == 44, 'Expected exactly 44 locked validation subjects.');
subjectCount = min(height(splitRows), floor(subjectLimit));
splitRows = splitRows(1:subjectCount, :);

methodIds = ["FILM", "MCAR", "RANF", "FSPAE"];
methodLabels = ["FiLM-SIREN D1/D2+notch E130 1/3 ensemble", ...
    "MCAR v3.5.1", "RANF", "FSP-AE"];
metricIds = ["FullSphereERB", "Contralateral25ERB", ...
    "ContralateralHighFrequency", "HorizontalILDMAE"];
perSubject = table();
metricLong = table();
qualityRows = table();

for subjectIndex = 1:subjectCount
    subjectLabel = splitRows.subject_id(subjectIndex);
    subjectNumber = sscanf(char(subjectLabel), 'P%d');
    sourceFile = fullfile(datasetRoot, 'subjects', char(subjectLabel), 'q26.h5');
    filmFile = fullfile(filmRoot, 'subjects', char(subjectLabel), 'prediction.h5');
    mcarFile = fullfile(mcarRoot, 'subjects', char(subjectLabel), 'prediction.h5');
    ranfFile = fullfile(ranfRoot, 'subjects', char(subjectLabel), 'prediction.sofa');
    fspFile = fullfile(fspRoot, 'subjects', char(subjectLabel), 'prediction.h5');
    sofaFile = fullfile(sofaRoot, sprintf( ...
        '%s_FreeFieldCompMinPhase_44kHz.sofa', subjectLabel));
    assert(isfile(sourceFile), 'Missing source HDF5: %s', sourceFile);
    assert(isfile(filmFile), 'Missing FiLM-SIREN prediction: %s', filmFile);
    assert(isfile(mcarFile), 'Missing MCAR prediction: %s', mcarFile);
    assert(isfile(ranfFile), 'Missing RANF prediction: %s', ranfFile);
    assert(isfile(fspFile), 'Missing FSP-AE prediction: %s', fspFile);
    assert(isfile(sofaFile), 'Missing %s SOFA: %s', 'val', sofaFile);
    assert(string(h5readatt(sourceFile, '/', 'split')) == "val", ...
        'Source split is not val for %s.', subjectLabel);
    assert(string(h5readatt(filmFile, '/', 'split')) == "val", ...
        'FiLM-SIREN split is not val for %s.', subjectLabel);
    assert(string(h5readatt(mcarFile, '/', 'split')) == "val", ...
        'MCAR split is not val for %s.', subjectLabel);
    assert(string(h5readatt(fspFile, '/', 'split')) == "val", ...
        'FSP-AE split is not val for %s.', subjectLabel);

    referenceDb = double(h5read(sourceFile, '/reference_logmag_db'));
    mcaDb = double(h5read(sourceFile, '/mca_logmag_db'));
    filmResidualDb = double(h5read(filmFile, '/predicted_residual_db'));
    mcarResidualDb = double(h5read(mcarFile, '/predicted_residual_db'));
    directionFeatures = double(h5read(sourceFile, '/direction_features'));
    frequencyHz = double(h5read(sourceFile, '/frequency_hz'));
    frequencyHz = frequencyHz(:);
    interpolationMask = logical(h5read( ...
        sourceFile, '/interpolation_evaluation_mask'));
    interpolationMask = interpolationMask(:);
    selectedPhase = double(h5read( ...
        sourceFile, '/strict_ild/mca_selected_phase_rad'));
    outsideReal = double(h5read( ...
        sourceFile, '/strict_ild/mca_outside_real'));
    outsideImag = double(h5read( ...
        sourceFile, '/strict_ild/mca_outside_imag'));
    selectedIndices = double(h5read(sourceFile, ...
        '/strict_ild/selected_bin_indices_zero_based')) + 1;
    selectedIndices = selectedIndices(:);
    outsideIndices = double(h5read(sourceFile, ...
        '/strict_ild/outside_bin_indices_zero_based')) + 1;
    outsideIndices = outsideIndices(:);
    referenceIldMetadata = double(h5read( ...
        sourceFile, '/strict_ild/reference_ild_db'));
    referenceIldMetadata = referenceIldMetadata(:);
    hrirLength = double(h5readatt(sourceFile, '/strict_ild', 'hrir_length'));
    samplingRateHz = double(h5readatt(sourceFile, '/', 'sampling_rate_hz'));

    expectedShape = [numel(frequencyHz), size(directionFeatures, 2), 2];
    assert(isequal(size(referenceDb), expectedShape), ...
        'Unexpected reference shape for %s.', subjectLabel);
    assert(isequal(size(filmResidualDb), expectedShape), ...
        'Unexpected FiLM-SIREN shape for %s.', subjectLabel);
    assert(isequal(size(mcarResidualDb), expectedShape), ...
        'Unexpected MCAR shape for %s.', subjectLabel);

    % FiLM-SIREN and MCAR: predicted residual dB added to corrected MCA dB,
    % reconstructed with the original MCA phase and outside bins.
    filmSelectedDb = mcaDb + filmResidualDb;
    mcarSelectedDb = mcaDb + mcarResidualDb;
    filmHrirLeft = reconstruct_hrir(filmSelectedDb(:, :, 1), ...
        selectedPhase(:, :, 1), outsideReal(:, :, 1), ...
        outsideImag(:, :, 1), selectedIndices, outsideIndices, hrirLength);
    filmHrirRight = reconstruct_hrir(filmSelectedDb(:, :, 2), ...
        selectedPhase(:, :, 2), outsideReal(:, :, 2), ...
        outsideImag(:, :, 2), selectedIndices, outsideIndices, hrirLength);
    mcarHrirLeft = reconstruct_hrir(mcarSelectedDb(:, :, 1), ...
        selectedPhase(:, :, 1), outsideReal(:, :, 1), ...
        outsideImag(:, :, 1), selectedIndices, outsideIndices, hrirLength);
    mcarHrirRight = reconstruct_hrir(mcarSelectedDb(:, :, 2), ...
        selectedPhase(:, :, 2), outsideReal(:, :, 2), ...
        outsideImag(:, :, 2), selectedIndices, outsideIndices, hrirLength);

    % RANF: SOFA HRIR with its own directions; verify ordering against the
    % HDF5 direction features, then compute magnitudes on the source grid.
    ranfSofa = SOFAload(ranfFile);
    ranfHrir = double(ranfSofa.Data.IR);
    assert(isequal(size(ranfHrir), [793, 2, 256]), ...
        'Unexpected RANF HRIR shape for %s.', subjectLabel);
    ranfPosition = double(ranfSofa.SourcePosition);
    azimuthDeg = directionFeatures(1, :).';
    elevationDeg = directionFeatures(2, :).';
    azimuthError = abs(mod(ranfPosition(:, 1) - azimuthDeg + 180, 360) - 180);
    elevationError = abs(ranfPosition(:, 2) - elevationDeg);
    assert(max(azimuthError) <= 1e-8 && max(elevationError) <= 1e-8, ...
        'RANF SOFA direction order mismatch for %s.', subjectLabel);
    ranfSpectra = complex(zeros(numel(frequencyHz), 793, 2));
    for ear = 1:2
        fullSpectrum = fft(squeeze(ranfHrir(:, ear, :)).', 1024, 1);
        ranfSpectra(:, :, ear) = fullSpectrum(selectedIndices, :);
    end
    ranfSelectedDb = 20 * log10(max(abs(ranfSpectra), 1e-10));

    % FSP-AE: direct HRIR plus magnitude on its own 512-bin grid. The FSP
    % frequency grid starts one bin before the source grid, so source
    % selected bin i maps to FSP bin i-1 (1-based).
    fspRaw = double(h5read(fspFile, '/predicted_magnitude_db'));
    fspMagnitudeDb = permute(fspRaw, [1, 3, 2]);
    fspSelectedDb = fspMagnitudeDb(selectedIndices - 1, :, :);
    fspHrirRaw = double(h5read(fspFile, '/predicted_hrir'));
    assert(isequal(size(fspHrirRaw), [793, 2, 256]), ...
        'Unexpected FSP-AE HRIR shape for %s.', subjectLabel);
    fspHrirLeft = squeeze(fspHrirRaw(:, 1, :)).';
    fspHrirRight = squeeze(fspHrirRaw(:, 2, :)).';

    assert(isequal(size(filmSelectedDb), expectedShape) && ...
        isequal(size(mcarSelectedDb), expectedShape) && ...
        isequal(size(fspSelectedDb), expectedShape) && ...
        isequal(size(ranfSelectedDb), expectedShape), ...
        'Four-method tensor shape mismatch for %s.', subjectLabel);
    assert(all(isfinite(filmSelectedDb), 'all') && ...
        all(isfinite(mcarSelectedDb), 'all') && ...
        all(isfinite(fspSelectedDb), 'all') && ...
        all(isfinite(ranfSelectedDb), 'all'), ...
        'Non-finite method magnitude for %s.', subjectLabel);

    % Reference HRIR and ILD validation.
    sofa = SOFAload(sofaFile);
    assert(isequal(size(sofa.Data.IR), [793, 2, hrirLength]), ...
        'Unexpected reference SOFA HRIR shape for %s.', subjectLabel);
    assert(abs(double(sofa.Data.SamplingRate) - samplingRateHz) < 1e-9, ...
        'Sampling-rate mismatch for %s.', subjectLabel);
    sofaAzimuth = mod(double(sofa.SourcePosition(:, 1)), 360);
    sofaElevation = double(sofa.SourcePosition(:, 2));
    azimuthErrorRef = abs(mod(sofaAzimuth - azimuthDeg + 180, 360) - 180);
    elevationErrorRef = abs(sofaElevation - elevationDeg);
    assert(max(azimuthErrorRef) <= 1e-7 && max(elevationErrorRef) <= 1e-7, ...
        'Reference SOFA/HDF5 direction ordering mismatch for %s.', subjectLabel);
    referenceHrirLeft = squeeze(double(sofa.Data.IR(:, 1, :))).';
    referenceHrirRight = squeeze(double(sofa.Data.IR(:, 2, :))).';
    referenceIldRaw = calculate_ild(referenceHrirLeft, referenceHrirRight);
    referenceIldError = max(abs(referenceIldRaw - referenceIldMetadata));
    assert(referenceIldError <= 2e-4, ...
        'Reference ILD metadata mismatch %.6g dB for %s.', ...
        referenceIldError, subjectLabel);

    % Metric regions.
    y = directionFeatures(4, :).';
    solidAngleWeight = directionFeatures(6, :).';
    fullMask = interpolationMask;
    leftContra25Mask = interpolationMask & great_circle_mask( ...
        azimuthDeg, elevationDeg, 270, 0, 25);
    rightContra25Mask = interpolationMask & great_circle_mask( ...
        azimuthDeg, elevationDeg, 90, 0, 25);
    leftContraHemisphereMask = interpolationMask & y < -1e-12;
    rightContraHemisphereMask = interpolationMask & y > 1e-12;
    horizontalMask = interpolationMask & abs(elevationDeg) <= 1e-9;
    assert(any(leftContra25Mask) && any(rightContra25Mask), ...
        'Contralateral 25-degree region is empty for %s.', subjectLabel);
    assert(any(horizontalMask), 'Horizontal validation mask is empty for %s.', subjectLabel);

    selectedDbCell = {filmSelectedDb, mcarSelectedDb, ranfSelectedDb, fspSelectedDb};
    hrirLeftCell = {filmHrirLeft, mcarHrirLeft, ranfHrir(:, :, 1).', fspHrirLeft};
    hrirRightCell = {filmHrirRight, mcarHrirRight, ranfHrir(:, :, 2).', fspHrirRight};
    metricValues = zeros(numel(methodIds), numel(metricIds));
    for methodIndex = 1:numel(methodIds)
        [erbErrorsLeft, erbFrequencyHz] = AKerbError( ...
            hrirLeftCell{methodIndex}(:, fullMask), ...
            referenceHrirLeft(:, fullMask), ...
            [50, samplingRateHz / 2], samplingRateHz);
        erbErrorsRight = AKerbError( ...
            hrirRightCell{methodIndex}(:, fullMask), ...
            referenceHrirRight(:, fullMask), ...
            [50, samplingRateHz / 2], samplingRateHz);
        fullWeights = normalized_weights(solidAngleWeight(fullMask));
        metricValues(methodIndex, 1) = 0.5 * (...
            mean(abs(erbErrorsLeft) * fullWeights) + ...
            mean(abs(erbErrorsRight) * fullWeights));

        leftWithinFull = leftContra25Mask(fullMask);
        rightWithinFull = rightContra25Mask(fullMask);
        leftWeights = normalized_weights(solidAngleWeight(leftContra25Mask));
        rightWeights = normalized_weights(solidAngleWeight(rightContra25Mask));
        metricValues(methodIndex, 2) = 0.5 * (...
            mean(abs(erbErrorsLeft(:, leftWithinFull)) * leftWeights) + ...
            mean(abs(erbErrorsRight(:, rightWithinFull)) * rightWeights));

        metricValues(methodIndex, 3) = 0.5 * (...
            high_frequency_error(selectedDbCell{methodIndex}(:, :, 1), ...
                referenceDb(:, :, 1), frequencyHz, ...
                leftContraHemisphereMask, solidAngleWeight) + ...
            high_frequency_error(selectedDbCell{methodIndex}(:, :, 2), ...
                referenceDb(:, :, 2), frequencyHz, ...
                rightContraHemisphereMask, solidAngleWeight));

        methodIld = calculate_ild(hrirLeftCell{methodIndex}, ...
            hrirRightCell{methodIndex});
        metricValues(methodIndex, 4) = mean(abs( ...
            methodIld(horizontalMask) - referenceIldMetadata(horizontalMask)));
    end
    assert(all(isfinite(metricValues), 'all'), ...
        'Non-finite metric for %s.', subjectLabel);

    row = table(subjectLabel, subjectNumber, ...
        metricValues(1, 1), metricValues(2, 1), metricValues(3, 1), metricValues(4, 1), ...
        metricValues(1, 2), metricValues(2, 2), metricValues(3, 2), metricValues(4, 2), ...
        metricValues(1, 3), metricValues(2, 3), metricValues(3, 3), metricValues(4, 3), ...
        metricValues(1, 4), metricValues(2, 4), metricValues(3, 4), metricValues(4, 4), ...
        'VariableNames', {'SubjectLabel', 'SubjectID', ...
        'FILMFullSphereERB_dB', 'MCARFullSphereERB_dB', ...
        'RANFFullSphereERB_dB', 'FSPAEFullSphereERB_dB', ...
        'FILMContralateral25ERB_dB', 'MCARContralateral25ERB_dB', ...
        'RANFContralateral25ERB_dB', 'FSPAEContralateral25ERB_dB', ...
        'FILMContralateralHighFrequency_dB', ...
        'MCARContralateralHighFrequency_dB', ...
        'RANFContralateralHighFrequency_dB', ...
        'FSPAEContralateralHighFrequency_dB', ...
        'FILMHorizontalILDMAE_dB', 'MCARHorizontalILDMAE_dB', ...
        'RANFHorizontalILDMAE_dB', 'FSPAEHorizontalILDMAE_dB'});
    perSubject = [perSubject; row]; %#ok<AGROW>
    for methodIndex = 1:numel(methodIds)
        for metricIndex = 1:numel(metricIds)
            metricLong = [metricLong; table(subjectLabel, subjectNumber, ...
                methodIds(methodIndex), methodLabels(methodIndex), ...
                metricIds(metricIndex), metricValues(methodIndex, metricIndex), ...
                'VariableNames', {'SubjectLabel', 'SubjectID', ...
                'Method', 'MethodLabel', 'Metric', 'Value_dB'})]; %#ok<AGROW>
        end
    end
    qualityRows = [qualityRows; table(subjectLabel, subjectNumber, ...
        referenceIldError, sum(fullMask), sum(horizontalMask), ...
        numel(erbFrequencyHz), ...
        'VariableNames', {'SubjectLabel', 'SubjectID', ...
        'ReferenceILDMetadataMaxError_dB', ...
        'InterpolationDirectionCount', 'HorizontalDirectionCount', ...
        'ERBBandCount'})]; %#ok<AGROW>
    fprintf('Strict four-method validation [%d/%d]: %s\n', ...
        subjectIndex, subjectCount, subjectLabel);
end

aggregate = make_aggregate(perSubject, metricIds, methodIds);
writetable(perSubject, fullfile(outputRoot, 'per_subject_metrics.csv'));
writetable(metricLong, fullfile(outputRoot, 'metric_long.csv'));
writetable(qualityRows, fullfile(outputRoot, 'quality_checks.csv'));
writetable(aggregate, fullfile(outputRoot, 'aggregate_metrics.csv'));
plot_aggregate_metrics(aggregate, methodIds, methodLabels, metricIds, figuresRoot);

summary = struct('schema_version', '1.0', 'status', 'completed', ...
    'split', 'val', 'subject_count', subjectCount, ...
    'test_subject_count_read', 0, ...
    'methods', methodIds, 'method_labels', methodLabels, ...
    'metrics', metricIds, ...
    'interpolation_direction_count', sum(fullMask), ...
    'horizontal_direction_count', sum(horizontalMask), ...
    'erb_function', 'AKerbError', ...
    'erb_frequency_range_hz', [50, samplingRateHz / 2], ...
    'contralateral_region_radius_degrees', 25, ...
    'contralateral_high_frequency_range_hz', [10000, 20000], ...
    'ild_region', 'horizontal-plane interpolation directions', ...
    'reconstruction', ...
        'FILM/MCAR: predicted magnitude with original MCA phase and outside bins; ' ...
        'RANF: SOFA HRIR; FSP-AE: predicted HRIR', ...
    'aggregate', table2struct(aggregate));
write_json(fullfile(outputRoot, 'summary.json'), summary);
fprintf('Strict four-method validation complete: %s\n', outputRoot);
clear restoreDirectory;
end

function hrir = reconstruct_hrir(selectedMagnitudeDb, selectedPhase, ...
        outsideReal, outsideImag, selectedIndices, outsideIndices, hrirLength)
directionCount = size(selectedMagnitudeDb, 2);
singleSidedCount = numel(selectedIndices) + numel(outsideIndices);
spectrum = complex(zeros(singleSidedCount, directionCount));
spectrum(selectedIndices, :) = 10 .^ (selectedMagnitudeDb / 20) .* ...
    exp(1i * selectedPhase);
spectrum(outsideIndices, :) = outsideReal + 1i * outsideImag;
hrirOversized = real(ifft(AKsingle2bothSidedSpectrum(spectrum)));
assert(size(hrirOversized, 1) >= hrirLength, ...
    'Reconstructed HRIR is shorter than expected.');
hrir = hrirOversized(1:hrirLength, :);
end

function ild = calculate_ild(leftHrir, rightHrir)
leftEnergy = sum(abs(leftHrir) .^ 2, 1);
rightEnergy = sum(abs(rightHrir) .^ 2, 1);
ild = (10 * log10(leftEnergy ./ rightEnergy)).';
end

function mask = great_circle_mask(azimuth, elevation, ...
        centerAzimuth, centerElevation, radiusDegrees)
mask = great_circle_distance(azimuth, elevation, ...
    centerAzimuth, centerElevation) <= radiusDegrees + 1e-10;
end

function distance = great_circle_distance(azimuth, elevation, ...
        centerAzimuth, centerElevation)
dotProduct = sind(elevation) .* sind(centerElevation) + ...
    cosd(elevation) .* cosd(centerElevation) .* ...
    cosd(azimuth - centerAzimuth);
distance = acosd(min(1, max(-1, dotProduct)));
end

function weights = normalized_weights(weights)
weights = weights(:);
assert(all(isfinite(weights)) && all(weights > 0), ...
    'Direction weights must be finite and positive.');
weights = weights / sum(weights);
end

function value = high_frequency_error(estimateDb, referenceDb, ...
        frequencyHz, directionMask, solidAngleWeight)
frequencyMask = frequencyHz > 10000 & frequencyHz <= 20000;
weights = normalized_weights(solidAngleWeight(directionMask));
errorDb = abs(estimateDb(frequencyMask, directionMask) - ...
    referenceDb(frequencyMask, directionMask));
value = mean(errorDb * weights);
end

function aggregate = make_aggregate(perSubject, metricIds, methodIds)
methodSuffixes = ["FullSphereERB_dB", "Contralateral25ERB_dB", ...
    "ContralateralHighFrequency_dB", "HorizontalILDMAE_dB"];
metricNames = ["Full ERB (dB)", "Contralateral 25-deg ERB (dB)", ...
    "Contralateral HF (dB)", "Horizontal ILD MAE (dB)"];
aggregate = table();
for metricIndex = 1:numel(metricIds)
    variables = methodIds + methodSuffixes(metricIndex);
    values = cell(1, numel(methodIds));
    for methodIndex = 1:numel(methodIds)
        values{methodIndex} = perSubject.(variables(methodIndex));
    end
    row = table(metricIds(metricIndex), metricNames(metricIndex), ...
        numel(values{1}), ...
        mean(values{1}), std(values{1}), ...
        mean(values{2}), std(values{2}), ...
        mean(values{3}), std(values{3}), ...
        mean(values{4}), std(values{4}), ...
        'VariableNames', {'Metric', 'MetricLabel', 'SubjectCount', ...
        'FILMMean_dB', 'FILMStd_dB', ...
        'MCARMean_dB', 'MCARStd_dB', ...
        'RANFMean_dB', 'RANFStd_dB', ...
        'FSPAEMean_dB', 'FSPAEStd_dB'});
    aggregate = [aggregate; row]; %#ok<AGROW>
end
end

function plot_aggregate_metrics(aggregate, methodIds, methodLabels, metricIds, figuresRoot)
figure('Visible', 'off');
colors = [0.00 0.45 0.74; 0.85 0.33 0.10; 0.47 0.67 0.19; 0.49 0.18 0.56];
barData = [aggregate.FILMMean_dB, aggregate.MCARMean_dB, ...
    aggregate.RANFMean_dB, aggregate.FSPAEMean_dB];
barHandle = bar(barData, 'grouped');
for methodIndex = 1:numel(methodIds)
    barHandle(methodIndex).FaceColor = colors(methodIndex, :);
end
xticklabels(metricIds);
ylabel('Error (dB)');
legend(methodLabels, 'Location', 'northoutside', 'Orientation', 'horizontal');
grid on;
title('SONICOM validation 44-subject four-method strict comparison');
saveas(gcf, fullfile(figuresRoot, 'four_method_aggregate_metrics.png'));
saveas(gcf, fullfile(figuresRoot, 'four_method_aggregate_metrics.pdf'));
close(gcf);
end

function write_json(path, value)
encoded = jsonencode(value);
fid = fopen(path, 'w', 'n', 'UTF-8');
fprintf(fid, '%s\n', encoded);
fclose(fid);
end
