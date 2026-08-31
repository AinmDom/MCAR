import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const root = process.cwd();
const outputDir = path.join(root, "outputs", "stage_e_complete_ten_method_comparison_v2");
const workbookPath = path.join(outputDir, "stage_e_complete_ten_method_comparison_v2.xlsx");
const input = await FileBlob.load(workbookPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const expected = ["Overview", "Method Registry", "Complete Table", "All Means", "Paired vs Bounded", "Availability", "Band ILD", "Spatial Map", "Efficiency", "Mechanism", "Correction Correlation"];
const actual = Array.from(workbook.worksheets).map((sheet) => sheet.name);
if (JSON.stringify(actual) !== JSON.stringify(expected)) throw new Error(`Sheet mismatch: ${actual}`);

const expectedShapes = {
  "Method Registry": [11, 5],
  "Complete Table": [25, 16],
  "All Means": [241, 15],
  "Paired vs Bounded": [208, 18],
  "Availability": [71, 6],
  "Band ILD": [351, 7],
  "Spatial Map": [7671, 8],
  "Efficiency": [3, 14],
  "Mechanism": [616, 11],
  "Correction Correlation": [45, 4],
};
let errorCount = 0;
for (const [sheetName, [rows, columns]] of Object.entries(expectedShapes)) {
  const values = workbook.worksheets.getItem(sheetName).getUsedRange().values;
  if (values.length !== rows || values[0].length !== columns) throw new Error(`${sheetName} shape ${values.length}x${values[0].length}`);
  for (const row of values) for (const value of row) if (typeof value === "string" && /#(REF!|DIV\/0!|VALUE!|NAME\?|N\/A)/.test(value)) errorCount += 1;
}
const registry = workbook.worksheets.getItem("Method Registry").getRange("A2:E11").values;
if (registry.length !== 10 || registry[9][1] !== "BOUNDED" || registry[9][2] !== "Bounded E25") throw new Error(`Registry tail invalid: ${JSON.stringify(registry[9])}`);
const completeHeaders = workbook.worksheets.getItem("Complete Table").getRange("G1:P1").values[0];
if (completeHeaders[0] !== "SH only" || completeHeaders[9] !== "Bounded E25") throw new Error(`Complete-table headers invalid: ${completeHeaders}`);
const overviewValues = workbook.worksheets.getItem("Overview").getRange("G5:G8").values.flat().map(Number);
if (JSON.stringify(overviewValues) !== JSON.stringify([10, 240, 240, 0])) throw new Error(`Overview checks invalid: ${overviewValues}`);
if (errorCount !== 0) throw new Error(`Formula error count: ${errorCount}`);
const report = { status: "verified", sheetCount: actual.length, expectedShapes, registryMethodCount: 10, overviewChecks: overviewValues, formulaErrors: errorCount };
await fs.writeFile(path.join(outputDir, "verification.json"), JSON.stringify(report, null, 2) + "\n", "utf8");
console.log(JSON.stringify(report));
