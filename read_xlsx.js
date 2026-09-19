const xlsx = require('xlsx');
const fs = require('fs');
const path = require('path');

const dataDir = "D:\\Workspace\\projects\\komoberi\\Tourism Dataset";
const files = fs.readdirSync(dataDir).filter(f => f.endsWith('.xlsx'));

files.forEach(file => {
    const filePath = path.join(dataDir, file);
    const workbook = xlsx.readFile(filePath);
    const sheetName = workbook.SheetNames[0];
    const worksheet = workbook.Sheets[sheetName];
    
    // Read max 5 rows
    const data = xlsx.utils.sheet_to_json(worksheet, { header: 1 });
    console.log(`\n=== File: ${file} ===`);
    console.log(`Rows: ${data.length}`);
    if (data.length > 0) {
        console.log(`Columns (${data[0].length}): ${data[0].join(', ')}`);
        console.log(`Data Types: ${data[0].map((col, idx) => {
            const val = data.length > 1 ? data[1][idx] : undefined;
            return typeof val;
        }).join(', ')}`);
    }
    
    // Show first 5 rows
    for (let i = 0; i < Math.min(6, data.length); i++) {
        console.log(`Row ${i}: ${JSON.stringify(data[i])}`);
    }
});
