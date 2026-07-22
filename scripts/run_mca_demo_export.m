%% Run the SUpDEq MCA demo and export every generated figure.
% The demo itself retains its original parameters: KU100, Lebedev Ns = 3,
% Lebedev Nd = 44, mc = inf, and a head radius of 0.0875 m.

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);
supdeqDir = fullfile(projectRoot, 'SUpDEq-master');
outputDir = fullfile(projectRoot, 'figures', 'mca_demo_ku100_ns3_nd44');

if ~isfolder(supdeqDir)
    error('SUpDEq directory was not found: %s', supdeqDir);
end
if ~isfolder(outputDir)
    mkdir(outputDir);
end

cd(supdeqDir);
supdeq_start;
close all;
run(fullfile(supdeqDir, 'supdeq_demo_MCA.m'));

figureHandles = findall(groot, 'Type', 'figure');
[~, sortOrder] = sort([figureHandles.Number]);
figureHandles = figureHandles(sortOrder);

figureNames = {
    '01_frontal_conventional_vs_mca_hrir'
    '02_contralateral_sh_vs_mca_hrir'
    '03_contralateral_conventional_vs_mca_hrir'
    '04_contralateral_sh_vs_reference_hrir'
    '05_contralateral_conventional_vs_reference_hrir'
    '06_contralateral_mca_vs_reference_hrir'
    '07_lsd_left_ear'
    '08_erb_magnitude_error_left_ear'};

for figureIndex = 1:numel(figureHandles)
    if figureIndex <= numel(figureNames)
        baseName = figureNames{figureIndex};
    else
        baseName = sprintf('%02d_demo_figure', figureIndex);
    end
    figure(figureHandles(figureIndex));
    drawnow;
    exportgraphics(figureHandles(figureIndex), fullfile(outputDir, [baseName '.png']), ...
        'Resolution', 200);
    savefig(figureHandles(figureIndex), fullfile(outputDir, [baseName '.fig']));
end

save(fullfile(outputDir, 'mca_demo_metrics.mat'), ...
    'lsd_sh', 'lsd_con', 'lsd_mca', 'erb_sh', 'erb_con', 'erb_mca', 'fc_erb', ...
    'Ns', 'Nd', 'headRadius', 'fs');

fprintf('Exported %d figures to: %s\n', numel(figureHandles), outputDir);
