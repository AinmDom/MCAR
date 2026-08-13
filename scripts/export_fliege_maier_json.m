function export_fliege_maier_json(outputPath)
%EXPORT_FLIEGE_MAIER_JSON Export project Fliege-Maier nodes for web viewer.
%   The source MAT file contains Cartesian unit vectors and quadrature
%   weights for spatial orders N=1,...,29. Weights are normalized per grid.

arguments
    outputPath (1,1) string = fullfile("configs", "data", ...
        "fliege_maier_nodes_1_29.json")
end

projectRoot = fileparts(fileparts(mfilename("fullpath")));
sourcePath = fullfile(projectRoot, "external", "SUpDEq", "materials", ...
    "nodes", "fliegeMaierNodes_1_30.mat");
if ~isfile(sourcePath)
    error("MCAR:FliegeSourceMissing", "Missing Fliege-Maier source: %s", sourcePath);
end

source = load(sourcePath, "fliegeNodes");
assert(iscell(source.fliegeNodes) && numel(source.fliegeNodes) >= 30, ...
    "Unexpected fliegeNodes container.");

grids = cell(29, 1);
for order = 1:29
    values = source.fliegeNodes{order + 1};
    expectedCount = (order + 1)^2;
    assert(isnumeric(values) && size(values, 1) == expectedCount && ...
        size(values, 2) >= 4, "Invalid Fliege-Maier grid N=%d.", order);

    xyz = values(:, 1:3);
    weights = values(:, 4);
    assert(max(abs(vecnorm(xyz, 2, 2) - 1)) < 1e-12, ...
        "Non-unit direction in Fliege-Maier grid N=%d.", order);
    % Remove source-table rounding drift while preserving directions.
    xyz = xyz ./ vecnorm(xyz, 2, 2);
    % Several published Fliege-Maier orders contain signed weights. Preserve
    % those values; they are quadrature coefficients, not cell areas.
    assert(all(isfinite(weights)) && abs(sum(weights)) > eps, ...
        "Invalid weight sum in Fliege-Maier grid N=%d.", order);
    weights = weights / sum(weights);
    assert(abs(sum(weights) - 1) < 1e-12, ...
        "Weights do not normalize for Fliege-Maier grid N=%d.", order);

    grid = struct;
    grid.order = order;
    grid.points = expectedCount;
    grid.values = [xyz, weights];
    grids{order} = grid;
end

payload = struct;
payload.source = "SUpDEq materials/nodes/fliegeMaierNodes_1_30.mat";
payload.description = "Fliege-Maier optimized spherical quadrature nodes";
payload.grids = grids;

absoluteOutput = fullfile(projectRoot, outputPath);
outputFolder = fileparts(absoluteOutput);
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end
writelines(jsonencode(payload), absoluteOutput, Encoding="UTF-8");
fprintf("Wrote %s with 29 Fliege-Maier grids.\n", absoluteOutput);
end
