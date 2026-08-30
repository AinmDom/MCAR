function generate_stage_e_external_comparison()
%GENERATE_STAGE_E_EXTERNAL_COMPARISON Build paper-facing Stage-E comparisons.

root = fileparts(fileparts(mfilename("fullpath")));
outputRoot = fullfile(root, "results", "sonicom_stage_e_external_comparison");
if isfolder(outputRoot)
    error("mcar:ExternalComparisonExists", "Refusing to overwrite %s", outputRoot);
end
mkdir(outputRoot);

primaryRoot = fullfile(root, "results", ...
    "sonicom_bounded_mcar_film_correction_final_e25_frozen_test_nine_method");
secondaryRoot = fullfile(root, "results", ...
    "sonicom_film_secondary_metrics_v1_validation");
deferredRoot = fullfile(root, "results", ...
    "sonicom_film_deferred_secondary_metrics_v1_validation");
externalRoot = fullfile(root, "results", ...
    "sonicom_film_secondary_baseline_extension_ranf_fsp_v1_validation");

primaryMeans = readtable(fullfile(primaryRoot, "aggregate.csv"), TextType="string");
primaryPairs = readtable(fullfile(primaryRoot, "paired_bootstrap.csv"), TextType="string");
secondaryMeansInternal = readtable(fullfile(secondaryRoot, "aggregate_metrics.csv"), TextType="string");
secondaryPairsInternal = readtable(fullfile(secondaryRoot, "paired_tail_risk.csv"), TextType="string");
deferredMeansInternal = readtable(fullfile(deferredRoot, "aggregate_endpoints.csv"), TextType="string");
deferredPairsInternal = readtable(fullfile(deferredRoot, "paired_vs_mcar.csv"), TextType="string");
secondaryMeansExternal = readtable(fullfile(externalRoot, "aggregate_metrics.csv"), TextType="string");
secondaryPairsExternal = readtable(fullfile(externalRoot, "paired_tail_risk.csv"), TextType="string");
deferredMeansExternal = readtable(fullfile(externalRoot, "aggregate_endpoints.csv"), TextType="string");
deferredPairsExternal = readtable(fullfile(externalRoot, "paired_deferred_tail_risk.csv"), TextType="string");

primary = makePrimary(primaryMeans, primaryPairs);
secondary = makeSupplementary(secondaryMeansInternal, secondaryMeansExternal, ...
    secondaryPairsInternal, secondaryPairsExternal, "Secondary", ...
    "Validation supplementary", "val");
deferred = makeSupplementary(deferredMeansInternal, deferredMeansExternal, ...
    deferredPairsInternal, deferredPairsExternal, "Deferred", ...
    "Exploratory validation", "val");
comparison = [primary; secondary; deferred];

assert(height(primary) == 12 && height(secondary) == 15 && height(deferred) == 18);
assert(all(isfinite(comparison.CandidateMean)) && all(isfinite(comparison.BaselineMean)));
assert(all(isfinite(comparison.MeanDifference)) && all(isfinite(comparison.CI95Lower)) && ...
    all(isfinite(comparison.CI95Upper)));

methodMeans = makeMethodMeans(primaryMeans, secondaryMeansInternal, ...
    secondaryMeansExternal, deferredMeansInternal, deferredMeansExternal);
scorecard = groupsummary(comparison, ["Tier", "BaselineLabel", "Status"]);
scorecard = renamevars(scorecard, "GroupCount", "EndpointCount");
overallScorecard = groupsummary(comparison, ["BaselineLabel", "Status"]);
overallScorecard = renamevars(overallScorecard, "GroupCount", "EndpointCount");
overallScorecard = addvars(overallScorecard, repmat("All", height(overallScorecard), 1), ...
    Before="BaselineLabel", NewVariableNames="Tier");
scorecard = [scorecard; overallScorecard];

availability = table( ...
    ["MCAR v3.5.1"; "RANF"; "FSP-AE"; "MCAR v3.5.1"; "RANF"; "FSP-AE"; ...
     "MCAR v3.5.1"; "RANF"; "FSP-AE"], ...
    ["Mechanism"; "Mechanism"; "Mechanism"; "Efficiency"; "Efficiency"; "Efficiency"; ...
     "Localization"; "Localization"; "Localization"], ...
    ["Not comparable: no FiLM gate"; "Not applicable: no common internal quantity"; ...
     "Not applicable: no common internal quantity"; ...
     "Not available under the frozen same-hardware protocol"; ...
     "Not available under the frozen same-hardware protocol"; ...
     "Not available under the frozen same-hardware protocol"; ...
     "Not run: immutable official dependency unavailable"; ...
     "Not run: immutable official dependency unavailable"; ...
     "Not run: immutable official dependency unavailable"], ...
    'VariableNames', ["BaselineLabel", "Category", "Status"]);

writetable(comparison, fullfile(outputRoot, "comparison_long.csv"));
writetable(methodMeans, fullfile(outputRoot, "method_means.csv"));
writetable(scorecard, fullfile(outputRoot, "scorecard.csv"));
writetable(availability, fullfile(outputRoot, "availability.csv"));
writetable(makePrimaryPaperTable(primaryMeans), fullfile(outputRoot, "paper_primary_table.csv"));

summary = struct;
summary.schema_version = "1.0";
summary.status = "completed";
summary.candidate = "Stage E Bounded E25 ensemble";
summary.baselines = ["MCAR v3.5.1", "RANF", "FSP-AE"];
summary.primary_split = "test";
summary.secondary_split = "val";
summary.subject_count_per_split = 44;
summary.comparison_rows = height(comparison);
summary.method_mean_rows = height(methodMeans);
summary.all_numeric_finite = true;
summary.new_test_subject_count_read = 0;
summary.upstream_candidate_test_subject_count_read = 44;
summary.primary_status_counts = statusCounts(primary);
summary.secondary_status_counts = statusCounts(secondary);
summary.deferred_status_counts = statusCounts(deferred);
summary.claim_boundary = [ ...
    "Primary metrics use the frozen engineering test, historically consumed at project level. ", ...
    "Secondary metrics are validation supplementary; deferred notch endpoints are exploratory. ", ...
    "No test data were read while generating this comparison."];
summary.sources = struct( ...
    primary=relative(root, primaryRoot), ...
    secondary_internal=relative(root, secondaryRoot), ...
    deferred_internal=relative(root, deferredRoot), ...
    external_extension=relative(root, externalRoot));
writelines(string(jsonencode(summary, PrettyPrint=true)), fullfile(outputRoot, "summary.json"));
writelines(makeMarkdown(primary, secondary, deferred, availability), ...
    fullfile(outputRoot, "STAGE_E_EXTERNAL_COMPARISON.md"));
end

function result = makePrimary(means, pairs)
candidateKey = "BOUNDED";
baselineKeys = ["MCARv351", "RANF", "FSPAE"];
baselineLabels = ["MCAR v3.5.1", "RANF", "FSP-AE"];
endpoints = ["FullSphereERB", "Contralateral25ERB", ...
    "ContralateralHighFrequency", "HorizontalILDMAE"];
rows = cell(numel(baselineKeys) * numel(endpoints), 18);
index = 0;
for baselineIndex = 1:numel(baselineKeys)
    baselineKey = baselineKeys(baselineIndex);
    for endpointIndex = 1:numel(endpoints)
        endpoint = endpoints(endpointIndex);
        pair = pairs(pairs.Baseline == baselineKey & pairs.Metric == endpoint, :);
        candidateMean = one(means.Mean_dB, means.Method == candidateKey & means.Metric == endpoint);
        baselineMean = one(means.Mean_dB, means.Method == baselineKey & means.Metric == endpoint);
        rank = candidateRank(means.Mean_dB, means.Method, means.Metric, endpoint);
        index = index + 1;
        rows(index, :) = row("Primary", "test", "Frozen engineering test", endpoint, ...
            "dB", baselineKey, baselineLabels(baselineIndex), candidateMean, baselineMean, ...
            pair.MeanDifference_dB, pair.CI95Lower_dB, pair.CI95Upper_dB, ...
            pair.CandidateWins, 44 - pair.CandidateWins, rank, pair.SubjectCount, ...
            "Lower is better", "paired_bootstrap.csv");
    end
end
result = cell2table(rows, VariableNames=columnNames());
result = finish(result);
end

function result = makeSupplementary(internalMeans, externalMeans, internalPairs, ...
        externalPairs, tier, evidence, split)
means = [internalMeans(internalMeans.Method == "BOUNDED" | internalMeans.Method == "MCAR", :); ...
    externalMeans];
baselineKeys = ["MCAR", "RANF", "FSPAE"];
baselineLabels = ["MCAR v3.5.1", "RANF", "FSP-AE"];
endpoints = unique(means.Endpoint, "stable");
endpoints(endpoints == "ReferenceNotchFraction") = [];
rows = cell(numel(baselineKeys) * numel(endpoints), 18);
index = 0;
for baselineIndex = 1:numel(baselineKeys)
    baselineKey = baselineKeys(baselineIndex);
    sourcePairs = externalPairs;
    if baselineKey == "MCAR"
        sourcePairs = internalPairs;
    end
    for endpointIndex = 1:numel(endpoints)
        endpoint = endpoints(endpointIndex);
        pair = sourcePairs(sourcePairs.Baseline == baselineKey & sourcePairs.Endpoint == endpoint, :);
        assert(height(pair) == 1, "Expected one paired row for %s/%s", baselineKey, endpoint);
        candidate = means(means.Method == "BOUNDED" & means.Endpoint == endpoint, :);
        baseline = means(means.Method == baselineKey & means.Endpoint == endpoint, :);
        assert(height(candidate) == 1 && height(baseline) == 1);
        rank = candidateRank(means.Mean, means.Method, means.Endpoint, endpoint);
        unit = endpointUnits(endpoint);
        if ismember("Unit", candidate.Properties.VariableNames)
            unit = candidate.Unit;
        end
        index = index + 1;
        rows(index, :) = row(tier, split, evidence, endpoint, unit, ...
            baselineKey, baselineLabels(baselineIndex), candidate.Mean, baseline.Mean, ...
            pair.MeanDifference, pair.Bootstrap95Lower, pair.Bootstrap95Upper, ...
            pair.Wins, pair.Losses, rank, pair.SubjectCount, "Lower is better", ...
            tier + " paired results");
    end
end
result = cell2table(rows, VariableNames=columnNames());
result = finish(result);
end

function result = finish(result)
result.RelativeDifference_pct = 100 .* result.MeanDifference ./ result.BaselineMean;
result.Status = repmat("Comparable", height(result), 1);
result.Status(result.CI95Upper < 0) = "Significantly better";
result.Status(result.CI95Lower > 0) = "Significantly worse";
result = movevars(result, ["RelativeDifference_pct", "Status"], After="MeanDifference");
end

function values = row(tier, split, evidence, endpoint, unit, baselineKey, baselineLabel, ...
        candidateMean, baselineMean, difference, lower, upper, wins, losses, rank, n, ...
        direction, source)
values = {tier, split, evidence, endpoint, unit, baselineKey, baselineLabel, ...
    candidateMean, baselineMean, difference, lower, upper, wins, losses, rank, n, ...
    direction, source};
end

function names = columnNames()
names = ["Tier", "Split", "EvidenceLevel", "Endpoint", "Unit", "BaselineKey", ...
    "BaselineLabel", "CandidateMean", "BaselineMean", "MeanDifference", ...
    "CI95Lower", "CI95Upper", "CandidateWins", "CandidateLosses", ...
    "CandidateRankOf4", "SubjectCount", "Direction", "Source"];
end

function value = one(values, selector)
value = values(selector);
assert(isscalar(value));
end

function rank = candidateRank(values, methods, endpoints, endpoint)
subset = values(endpoints == endpoint & ismember(methods, ["BOUNDED", "MCAR", "MCARv351", "RANF", "FSPAE"]));
assert(numel(subset) == 4);
candidate = values(endpoints == endpoint & methods == "BOUNDED");
rank = 1 + sum(subset < candidate);
end

function means = makeMethodMeans(primary, secondaryInternal, secondaryExternal, ...
        deferredInternal, deferredExternal)
p = primary(ismember(primary.Method, ["BOUNDED", "MCARv351", "RANF", "FSPAE"]), ...
    ["Method", "MethodLabel", "Metric", "Mean_dB", "Std_dB", "SubjectCount"]);
p = renamevars(p, ["Metric", "Mean_dB", "Std_dB"], ["Endpoint", "Mean", "SampleStd"]);
p.Tier = repmat("Primary", height(p), 1); p.Split = repmat("test", height(p), 1);
p.EvidenceLevel = repmat("Frozen engineering test", height(p), 1); p.Unit = repmat("dB", height(p), 1);
p.Method(p.Method == "MCARv351") = "MCAR";
s = [secondaryInternal(ismember(secondaryInternal.Method, ["BOUNDED", "MCAR"]), :); secondaryExternal];
s.Tier = repmat("Secondary", height(s), 1); s.Split = repmat("val", height(s), 1);
s.EvidenceLevel = repmat("Validation supplementary", height(s), 1);
d = [deferredInternal(ismember(deferredInternal.Method, ["BOUNDED", "MCAR"]), :); deferredExternal];
d.MethodLabel = methodLabels(d.Method);
d.Unit = endpointUnits(d.Endpoint);
d.Tier = repmat("Deferred", height(d), 1); d.Split = repmat("val", height(d), 1);
d.EvidenceLevel = repmat("Exploratory validation", height(d), 1);
variables = ["Tier", "Split", "EvidenceLevel", "Method", "MethodLabel", "Endpoint", ...
    "Unit", "SubjectCount", "Mean", "SampleStd", "Bootstrap95Lower", "Bootstrap95Upper"];
p.Bootstrap95Lower = nan(height(p), 1); p.Bootstrap95Upper = nan(height(p), 1);
means = [p(:, variables); s(:, variables); d(:, variables)];
end

function labels = methodLabels(methods)
labels = methods;
labels(methods == "BOUNDED") = "Stage E Bounded E25";
labels(methods == "MCAR") = "MCAR v3.5.1";
labels(methods == "RANF") = "RANF";
labels(methods == "FSPAE") = "FSP-AE";
end

function units = endpointUnits(endpoints)
units = repmat("fraction", size(endpoints));
units(contains(endpoints, "_Hz")) = "Hz";
units(contains(endpoints, "_us")) = "us";
end

function result = makePrimaryPaperTable(means)
methods = ["BOUNDED", "MCARv351", "RANF", "FSPAE"];
labels = ["Stage E Bounded E25", "MCAR v3.5.1", "RANF", "FSP-AE"];
endpoints = ["FullSphereERB", "Contralateral25ERB", ...
    "ContralateralHighFrequency", "HorizontalILDMAE"];
data = zeros(4, 4);
for i = 1:4
    for j = 1:4
        data(i, j) = one(means.Mean_dB, means.Method == methods(i) & means.Metric == endpoints(j));
    end
end
result = array2table(data, VariableNames=endpoints);
result = addvars(result, methods', labels', Before=1, ...
    NewVariableNames=["Method", "MethodLabel"]);
end

function counts = statusCounts(rows)
counts = struct;
counts.significantly_better = nnz(rows.Status == "Significantly better");
counts.comparable = nnz(rows.Status == "Comparable");
counts.significantly_worse = nnz(rows.Status == "Significantly worse");
end

function lines = makeMarkdown(primary, secondary, deferred, availability)
lines = [ ...
    "# Stage E external horizontal comparison", "", ...
    "Candidate: frozen Stage E Bounded MCAR + FiLM-SIREN correction E25 1/3 ensemble.", "", ...
    "## Evidence boundary", "", ...
    "- Primary four metrics use the 44-subject frozen engineering test. The project test has historical consumption, so this is not presented as a study-wide untouched confirmation.", ...
    "- Secondary spectral/ILD metrics use the 44-subject validation split and are supplementary.", ...
    "- Deferred ITD and dominant-notch endpoints use validation; dominant-notch results are exploratory.", ...
    "- Every difference is Stage E minus baseline; lower is better. No data HDF5 were read while generating this report.", "", ...
    "## Primary frozen-test comparison", "", ...
    "| Baseline | Metric | Stage E | Baseline | Difference | 95% CI | Wins | Decision | Rank/4 |", ...
    "|---|---|---:|---:|---:|---:|---:|---|---:|"];
lines = lines(:);
lines = [lines; markdownRows(primary)];
lines = [lines; ""; "## Validation supplementary comparison"; ""; ...
    "| Baseline | Metric | Stage E | Baseline | Difference | 95% CI | Wins | Decision | Rank/4 |"; ...
    "|---|---|---:|---:|---:|---:|---:|---|---:|"; markdownRows(secondary)];
lines = [lines; ""; "## Exploratory/deferred validation comparison"; ""; ...
    "| Baseline | Metric | Stage E | Baseline | Difference | 95% CI | Wins | Decision | Rank/4 |"; ...
    "|---|---|---:|---:|---:|---:|---:|---|---:|"; markdownRows(deferred)];
lines = [lines; ""; "## Comparability limitations"; ""; ...
    "| Baseline | Category | Status |"; "|---|---|---|"; ...
    compose("| %s | %s | %s |", availability.BaselineLabel, availability.Category, availability.Status); ""; ...
    "## Paper-facing conclusion"; ""; ...
    "Stage E is strongest on the two broad ERB endpoints and improves MCAR on the three primary spectral endpoints. Its primary horizontal ILD is statistically comparable to MCAR, RANF, and FSP-AE, with a slightly worse mean than MCAR. RANF and especially FSP-AE retain advantages in high-frequency spectral-detail metrics; FSP-AE is decisively better on primary contralateral HF. Stage E improves ERB-band ILD over all three baselines. ITD is comparable to MCAR and FSP-AE and better than RANF. Dominant-notch location metrics are exploratory and favor the external baselines over Stage E."];
end

function lines = markdownRows(rows)
lines = compose("| %s | %s | %.6g | %.6g | %+.6g | [%+.6g, %+.6g] | %d/%d | %s | %d |", ...
    rows.BaselineLabel, rows.Endpoint, rows.CandidateMean, rows.BaselineMean, ...
    rows.MeanDifference, rows.CI95Lower, rows.CI95Upper, rows.CandidateWins, ...
    rows.SubjectCount, rows.Status, rows.CandidateRankOf4);
end

function path = relative(root, absolute)
path = erase(string(absolute), string(root) + filesep);
path = replace(path, filesep, "/");
end
