import { add, substract, multiply, divide, pi } from './math.js';

document.getElementById('addBtn').addEventListener('click', () => calculate(add));
document.getElementById('substractBtn').addEventListener('click', () => calculate(substract));
document.getElementById('multiplyBtn').addEventListener('click', () => calculate(multiply));
document.getElementById('divideBtn').addEventListener('click', () => calculate(divide));


function calculate(operation) {
    const a = parseInt(document.getElementById('num1').value);
    const b = parseInt(document.getElementById('num2').value);
    const resultElement = document.getElementById('result');

    if (isNaN(num1) || isNaN(num2)) {
        resultElement.textContent = "Please enter valid numbers.";
        return;
    }

    const result = operation(a,b);
    resultElement.textContent = `Result: ${result}`;
}