import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = process.cwd();
const resultDir = path.join(root, "results", "sonicom_complete_ten_method_comparison_v2");
const outputDir = path.join(root, "outputs", "stage_e_complete_ten_method_comparison_v2");
const previewDir = path.join(outputDir, "previews");
await fs.mkdir(previewDir, { recursive: true });

const workbook = Workbook.create();
const overview = workbook.worksheets.add("Overview");

const imports = [
  ["Method Registry", "method_registry.csv"],
  ["Complete Table", "paper_complete_comparison_wide.csv"],
  ["All Means", "method_endpoint_means.csv"],
  ["Paired vs Bounded", "paired_vs_bounded.csv"],
  ["Availability", "availability_matrix.csv"],
  ["Band ILD", "band_ild_profile_aggregate.csv"],
  ["Spatial Map", "spatial_direction_map.csv"],
  ["Efficiency", "bounded_efficiency_summary.csv"],
  ["Mechanism", "bounded_mechanism_aggregate.csv"],
  ["Correction Correlation", "bounded_correction_benefit_spearman.csv"],
];

for (const [sheetName, fileName] of imports) {
  const csvText = await fs.readFile(path.join(resultDir, fileName), "utf8");
  const imported = await Workbook.fromCSV(csvText, { sheetName });
  const importedValues = imported.worksheets.getItem(sheetName).getUsedRange().values;
  const target = workbook.worksheets.add(sheetName);
  target.getRangeByIndexes(0, 0, importedValues.length, importedValues[0].length).values = importedValues;
}

const navy = "#17365D";
const teal = "#0F766E";
const paleBlue = "#DCE6F1";
const paleGreen = "#E2F0D9";
const paleYellow = "#FFF2CC";
const paleRed = "#FCE4D6";
const grid = "#D9E2F3";
const white = "#FFFFFF";

function columnLetter(index) {
  let value = index + 1;
  let result = "";
  while (value > 0) {
    value -= 1;
    result = String.fromCharCode(65 + (value % 26)) + result;
    value = Math.floor(value / 26);
  }
  return result;
}

function setupDataSheet(sheet, tableName, widths = {}) {
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  const used = sheet.getUsedRange();
  const values = used.values;
  const rowCount = values.length;
  const colCount = values[0].length;
  const end = `${columnLetter(colCount - 1)}${rowCount}`;
  sheet.tables.add(`A1:${end}`, true, tableName).style = "TableStyleMedium2";
  sheet.getRange(`A1:${columnLetter(colCount - 1)}1`).format = {
    fill: navy,
    font: { bold: true, color: white },
    wrapText: true,
    verticalAlignment: "center",
    borders: { preset: "all", style: "thin", color: grid },
  };
  sheet.getRange(`A2:${end}`).format.borders = { preset: "all", style: "thin", color: grid };
  sheet.getRange(`A1:${end}`).format.verticalAlignment = "center";
  sheet.getRange(`A1:${end}`).format.autofitRows();
  for (let index = 0; index < colCount; index += 1) {
    const header = String(values[0][index]);
    const width = widths[header] ?? (header.includes("Reason") || header.includes("Source") ? 34 : 16);
    sheet.getRange(`${columnLetter(index)}:${columnLetter(index)}`).format.columnWidth = width;
  }
  return { values, rowCount, colCount, end };
}

const registry = workbook.worksheets.getItem("Method Registry");
setupDataSheet(registry, "MethodRegistryTable", { MethodLabel: 24, Family: 27, Method: 18 });
registry.freezePanes.freezeColumns(2);

const complete = workbook.worksheets.getItem("Complete Table");
const completeMeta = setupDataSheet(complete, "CompleteComparisonTable", {
  EvidenceTier: 30,
  EvidenceLevel: 31,
  Endpoint: 34,
  Direction: 18,
  "SUpDEq Barycentric": 22,
});
complete.freezePanes.freezeColumns(6);
complete.getRange(`G2:P${completeMeta.rowCount}`).format.numberFormat = "0.000";
complete.getRange(`G2:P${completeMeta.rowCount}`).conditionalFormats.add("containsText", {
  text: "NOT RUN",
  format: { fill: paleYellow, font: { color: "#7F6000", italic: true } },
});
complete.getRange(`G2:P${completeMeta.rowCount}`).conditionalFormats.add("colorScale", {
  thresholds: ["min", "50%", "max"],
  colors: ["#E2F0D9", "#FFF2CC", "#F8CBAD"],
});

const means = workbook.worksheets.getItem("All Means");
const meansMeta = setupDataSheet(means, "AllMeansTable", {
  EvidenceTier: 30,
  EvidenceLevel: 30,
  Endpoint: 34,
  MethodLabel: 24,
  Reason: 48,
});
means.freezePanes.freezeColumns(9);
means.getRange(`M2:N${meansMeta.rowCount}`).format.numberFormat = "0.000";
means.getRange(`J2:J${meansMeta.rowCount}`).conditionalFormats.add("containsText", {
  text: "AVAILABLE",
  format: { fill: paleGreen, font: { color: "#375623", bold: true } },
});
means.getRange(`J2:J${meansMeta.rowCount}`).conditionalFormats.add("containsText", {
  text: "NOT RUN",
  format: { fill: paleYellow, font: { color: "#7F6000", bold: true } },
});

const paired = workbook.worksheets.getItem("Paired vs Bounded");
const pairedMeta = setupDataSheet(paired, "PairedComparisonTable", {
  EvidenceTier: 30,
  Endpoint: 34,
  CandidateLabel: 20,
  BaselineLabel: 24,
  DifferenceDefinition: 46,
});
paired.freezePanes.freezeColumns(9);
paired.getRange(`J2:L${pairedMeta.rowCount}`).format.numberFormat = "0.000";
paired.getRange(`J2:J${pairedMeta.rowCount}`).conditionalFormats.add("cellIs", {
  operator: "lessThan",
  formula: 0,
  format: { fill: paleGreen, font: { color: "#375623" } },
});
paired.getRange(`J2:J${pairedMeta.rowCount}`).conditionalFormats.add("cellIs", {
  operator: "greaterThan",
  formula: 0,
  format: { fill: paleRed, font: { color: "#9C0006" } },
});

const availability = workbook.worksheets.getItem("Availability");
const availabilityMeta = setupDataSheet(availability, "AvailabilityTable", {
  MethodLabel: 24,
  MetricGroup: 34,
  Status: 20,
  Reason: 52,
});
availability.freezePanes.freezeColumns(3);
availability.getRange(`E2:E${availabilityMeta.rowCount}`).conditionalFormats.add("containsText", {
  text: "AVAILABLE",
  format: { fill: paleGreen, font: { color: "#375623", bold: true } },
});
availability.getRange(`E2:E${availabilityMeta.rowCount}`).conditionalFormats.add("containsText", {
  text: "NOT RUN",
  format: { fill: paleYellow, font: { color: "#7F6000" } },
});
availability.getRange(`E2:E${availabilityMeta.rowCount}`).conditionalFormats.add("containsText", {
  text: "NOT AVAILABLE",
  format: { fill: paleRed, font: { color: "#9C0006" } },
});

const band = workbook.worksheets.getItem("Band ILD");
const bandMeta = setupDataSheet(band, "BandILDTable", { MethodLabel: 24, BandCenter_Hz: 20 });
band.freezePanes.freezeColumns(2);
band.getRange(`D2:G${bandMeta.rowCount}`).format.numberFormat = "0.000";

const spatial = workbook.worksheets.getItem("Spatial Map");
const spatialMeta = setupDataSheet(spatial, "SpatialMapTable", { MethodLabel: 24 });
spatial.freezePanes.freezeColumns(2);
spatial.getRange(`C2:H${spatialMeta.rowCount}`).format.numberFormat = "0.000";

const efficiency = workbook.worksheets.getItem("Efficiency");
const efficiencyMeta = setupDataSheet(efficiency, "EfficiencyTable", { Deployment: 25 });
efficiency.getRange(`B2:N${efficiencyMeta.rowCount}`).format.numberFormat = "0.000";

const mechanism = workbook.worksheets.getItem("Mechanism");
const mechanismMeta = setupDataSheet(mechanism, "MechanismTable", { MethodLabel: 24, Entity: 22, Quantity: 22, Region: 22, Statistic: 26 });
mechanism.freezePanes.freezeColumns(4);
mechanism.getRange(`H2:K${mechanismMeta.rowCount}`).format.numberFormat = "0.000";

const correlation = workbook.worksheets.getItem("Correction Correlation");
const correlationMeta = setupDataSheet(correlation, "CorrectionCorrelationTable", { MethodLabel: 24, SubjectLabel: 18 });
correlation.freezePanes.freezeColumns(2);
correlation.getRange(`D2:D${correlationMeta.rowCount}`).format.numberFormat = "0.000";

overview.showGridLines = false;
overview.freezePanes.freezeRows(5);
overview.getRange("A1:P1").merge();
overview.getRange("A1").values = [["Complete Ten-Method Horizontal Comparison"]];
overview.getRange("A1:P1").format = {
  fill: navy,
  font: { bold: true, color: white, size: 18 },
  rowHeight: 34,
  horizontalAlignment: "left",
  verticalAlignment: "center",
};
overview.getRange("A2:P2").merge();
overview.getRange("A2").values = [["Registered order: SH only · SUpDEq SH · SUpDEq NN · SUpDEq Barycentric · MCA · MCAR v3.5.1 · FSP-AE · RANF · Hybrid E190 · Bounded E25"]];
overview.getRange("A2:P2").format = { fill: paleBlue, font: { color: navy, italic: true }, wrapText: true, rowHeight: 32 };
overview.getRange("A4:D4").values = [["Coverage summary", "Available", "Total", "Boundary"]];
overview.getRange("A5:A8").values = [["Primary validation"], ["Frozen engineering test"], ["Secondary validation"], ["Deferred validation"]];
overview.getRange("B5:B8").values = [[10], [10], [10], [10]];
overview.getRange("C5:C8").values = [[10], [10], [10], [10]];
overview.getRange("D5:D8").values = [["44-subject validation"], ["Frozen engineering test"], ["Supplementary validation"], ["Exploratory validation"]];
overview.getRange("A4:D8").format.borders = { preset: "all", style: "thin", color: grid };
overview.getRange("A4:D4").format = { fill: teal, font: { bold: true, color: white } };
overview.getRange("B5:C8").format.numberFormat = "0";
overview.getRange("F4:H4").values = [["Integrity check", "Value", "Meaning"]];
overview.getRange("F5:F8").values = [["Registered methods"], ["Method-endpoint cells"], ["Available numeric cells"], ["New test subjects read"]];
overview.getRange("G5:G8").formulas = [["=COUNTA('Method Registry'!$B$2:$B$11)"], ["=COUNTA('All Means'!$A$2:$A$241)"], ["=COUNT('All Means'!$M$2:$M$241)"], ["=0"]];
overview.getRange("H5:H8").values = [["Canonical set"], ["All tiers including structural missingness"], ["Finite upstream means"], ["Consolidation-only; no raw test read"]];
overview.getRange("F4:H8").format.borders = { preset: "all", style: "thin", color: grid };
overview.getRange("F4:H4").format = { fill: teal, font: { bold: true, color: white } };
overview.getRange("G5:G8").format.numberFormat = "0";
overview.getRange("A10:P10").merge();
overview.getRange("A10").values = [["Evidence rule: validation, historical frozen engineering test, supplementary validation, and exploratory validation are never pooled into one rank. NOT RUN / NOT AVAILABLE / NOT APPLICABLE are structural states, not zeros."]];
overview.getRange("A10:P10").format = { fill: paleYellow, font: { color: "#7F6000", bold: true }, wrapText: true, rowHeight: 42 };
overview.getRange("A12:C12").values = [["Method", "Validation FullSphere ERB", "Frozen-test FullSphere ERB"]];
overview.getRange("A13:A22").values = [
  ["SH only"], ["SUpDEq SH"], ["SUpDEq NN"], ["SUpDEq Barycentric"], ["MCA"],
  ["MCAR v3.5.1"], ["FSP-AE"], ["RANF"], ["Hybrid E190"], ["Bounded E25"],
];
const chartFormulaRows = [];
for (let index = 0; index < 10; index += 1) {
  const sourceColumn = columnLetter(6 + index);
  chartFormulaRows.push([`='Complete Table'!${sourceColumn}$2`, `='Complete Table'!${sourceColumn}$6`]);
}
overview.getRange("B13:C22").formulas = chartFormulaRows;
overview.getRange("A12:C22").format.borders = { preset: "all", style: "thin", color: grid };
overview.getRange("A12:C12").format = { fill: teal, font: { bold: true, color: white }, wrapText: true };
overview.getRange("B13:C22").format.numberFormat = "0.000";
const chart = overview.charts.add("bar", overview.getRange("A12:C22"));
chart.title = "Full-sphere ERB: validation and frozen engineering test";
chart.hasLegend = true;
chart.xAxis = { axisType: "textAxis", textStyle: { fontSize: 9 } };
chart.yAxis = { numberFormatCode: "0.0", min: 0 };
chart.setPosition("F12", "P29");

overview.getRange("A:A").format.columnWidth = 28;
overview.getRange("B:C").format.columnWidth = 19;
overview.getRange("D:D").format.columnWidth = 42;
overview.getRange("F:F").format.columnWidth = 28;
overview.getRange("G:G").format.columnWidth = 16;
overview.getRange("H:H").format.columnWidth = 42;

const sheetNames = ["Overview", ...imports.map(([name]) => name)];
let formulaErrors = 0;
for (const sheetName of sheetNames) {
  const sheet = workbook.worksheets.getItem(sheetName);
  const values = sheet.getUsedRange().values;
  for (const row of values) {
    for (const value of row) {
      if (typeof value === "string" && /#(REF!|DIV\/0!|VALUE!|NAME\?|N\/A)/.test(value)) formulaErrors += 1;
    }
  }
  const preview = await workbook.render({
    sheetName,
    range: sheetName === "Overview" ? "A1:P29" : `A1:${columnLetter(Math.min(values[0].length, 16) - 1)}${Math.min(values.length, 35)}`,
    scale: 0.8,
    format: "png",
  });
  await fs.writeFile(path.join(previewDir, `${sheetName.replaceAll(" ", "_")}.png`), new Uint8Array(await preview.arrayBuffer()));
}

if (formulaErrors !== 0) throw new Error(`Formula error scan found ${formulaErrors} errors`);
const inspected = await workbook.inspect({ kind: "sheet", include: "id,name", maxChars: 6000 });
await fs.writeFile(path.join(outputDir, "workbook_inspection.txt"), String(inspected.ndjson ?? inspected), "utf8");
await fs.writeFile(path.join(outputDir, "formula_error_count.txt"), `${formulaErrors}\n`, "utf8");
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(outputDir, "stage_e_complete_ten_method_comparison_v2.xlsx"));
console.log(JSON.stringify({ sheetNames, formulaErrors, output: path.join(outputDir, "stage_e_complete_ten_method_comparison_v2.xlsx") }));
