/* 
 * toy
 * by happy
 */
/*
 * TODO
 * get the basics done (*)
 * finish the nn (*)
 * clean (*)
 * NODEjs compatibility (*)
*/
//creating the NeuralNetwork function
/* 
 * NeuralNetwork
 * The main NeuralNetwork constructor
 * Example: let brain = new NeuralNetwork(2, 2, 1)
 */
class NeuralNetwork {
    constructor(NumInputs, NumHidden, NumOutput) {
        //Define the nodes
        this.inputNodes = NumInputs
        this.hiddenNodes = NumHidden
        this.outputNodes = NumOutput
        //Define the weights
        this.weights_ih = new Matrix(this.hiddenNodes, this.inputNodes)
        this.weights_ho = new Matrix(this.outputNodes, this.hiddenNodes)
        this.weights_ih.randomize()
        this.weights_ho.randomize()
        //Define bias
        this.bias_h = new Matrix(this.hiddenNodes, 1)
        this.bias_o = new Matrix(this.outputNodes, 1)
        this.bias_h.randomize()
        this.bias_o.randomize()
        this.learning_rate = 0.1
    }
    
    predict(input) {
        if (input.length == this.inputNodes) {
            //generating the hidden outputs
            let inputs = Matrix.fromArray(input)
            let hidden = Matrix.multiply(this.weights_ih, inputs)
            hidden.add(this.bias_h)
            // activation function!!
            hidden.map(sigmoid)
            //generating the output's output
            let output = Matrix.multiply(this.weights_ho, hidden)
            output.add(this.bias_o)
            output.map(sigmoid)
            //return the output
            return output.toArray()
        }else {
            console.error("The number of input nodes must match the number of inputs you put into feedForward()")
            return undefined
        }
    }
    /*
     * Train
     * The training function for the NeuralNetwork
     * example:
     * let brain = new NeuralNetwork(2, 2, 1)
     * var inputs = [1, 0]
     * var targets = [1]
     * brain.train(inputs, targets)
    */
    train(input_array, target_array) {
        //generating the hidden outputs
        let inputs = Matrix.fromArray(input_array)
        let hidden = Matrix.multiply(this.weights_ih, inputs)
        hidden.add(this.bias_h)
        // activation function!!
        hidden.map(sigmoid)
        //generating the output's output
        let outputs = Matrix.multiply(this.weights_ho, hidden)
        outputs.add(this.bias_o)
        outputs.map(sigmoid)
        //convert array to matrix
        let targets = Matrix.fromArray(target_array)
        //calculate the error
        //Error = targets - outputs
        let output_error = Matrix.subtract(targets, outputs) 
        // Calc gradient
        //let gradient = outputs * (1 - outputs)
        let gradients = Matrix.map(outputs, dsigmoid)
        //gradients.map(dsigmoid)
        gradients.multiply(output_error)
        gradients.multiply(this.learning_rate)
         // Calc deltas
        let hidden_T = Matrix.transpose(hidden)
        let weight_ho_deltas = Matrix.multiply(gradients, hidden_T)
        // Adjust the weights by deltas
        this.weights_ho.add(weight_ho_deltas)
        // Adjust the bias by its deltas (which is just the gradients)
        this.bias_o.add(gradients)
        //calculate the hidden layer errors
        let who_t = Matrix.transpose(this.weights_ho)
        let hidden_errors = Matrix.multiply(who_t, output_error)
        // Calc hidden gradient
        let hidden_gradient = Matrix.map(hidden, dsigmoid)
        hidden_gradient.multiply(hidden_errors)
        hidden_gradient.multiply(this.learning_rate)
        // Calc input->hidden deltas
        let inputs_T = Matrix.transpose(inputs)
        let weight_ih_deltas = Matrix.multiply(hidden_gradient, inputs_T)
        
        this.weights_ih.add(weight_ih_deltas)
        // Adjust the bias by its deltas (which is just the gradients)
        this.bias_h.add(hidden_gradient)
        /*outputs.print()
        targets.print()
        output_error.print()
        who_t.print()
        hidden_errors.print()*/
    }

    setLearningRate(learning_rate = 0.1) {
        this.learning_rate = learning_rate;
    }
    
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x))
}

function dsigmoid(y) {
    //derivative of sigmoid
    //return sigmoid(x) * (1 - sigmoid(x))
    return y * (1 - y)
}
// let m = new Matrix(3,2);
class Matrix {
    constructor(rows, cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = Array(this.rows).fill().map(() => Array(this.cols).fill(0));
    }
    
    copy() {
        let m = new Matrix(this.rows, this.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                m.data[i][j] = this.data[i][j];
            }
        }
        return m;
    }
    
    static fromArray(arr) {
        return new Matrix(arr.length, 1).map((e, i) => arr[i]);
    }
    
    static subtract(a, b) {
        if (a.rows !== b.rows || a.cols !== b.cols) {
            console.error('Columns and Rows of A must match Columns and Rows of B.');
            return;
        }
        // Return a new Matrix a-b
        return new Matrix(a.rows, a.cols).map((_, i, j) => a.data[i][j] - b.data[i][j]);
    }
    
    toArray() {
        let arr = [];
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                arr.push(this.data[i][j]);
            }
        }
        return arr;
    }
    
    randomize() {
        return this.map(e => Math.random() * 2 - 1);
    }
    
    add(n) {
        if (n instanceof Matrix) {
            if (this.rows !== n.rows || this.cols !== n.cols) {
                console.error('Columns and Rows of A must match Columns and Rows of B.');
                return;
            }
            return this.map((e, i, j) => e + n.data[i][j]);
        } else {
            return this.map(e => e + n);
        }
    }
    
    static transpose(matrix) {
        return new Matrix(matrix.cols, matrix.rows).map((_, i, j) => matrix.data[j][i]);
    }
    
    static multiply(a, b) {
        // Matrix product
        if (a.cols !== b.rows) {
            console.error('Columns of A must match rows of B.');
            return;
        }
        return new Matrix(a.rows, b.cols).map((e, i, j) => {
            // Dot product of values in col
            let sum = 0;
            for (let k = 0; k < a.cols; k++) {
                sum += a.data[i][k] * b.data[k][j];
            }
            return sum;
        });
    }
    
    multiply(n) {
        if (n instanceof Matrix) {
            if (this.rows !== n.rows || this.cols !== n.cols) {
                console.error('Columns and Rows of A must match Columns and Rows of B.');
                return;
            }
            // hadamard product
            return this.map((e, i, j) => e * n.data[i][j]);
        } else {
            // Scalar product
            return this.map(e => e * n);
        }
    }
    
    map(func) {
        // Apply a function to every element of matrix
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                let val = this.data[i][j];
                this.data[i][j] = func(val, i, j);
            }
        }
        return this;
    }
    
    static map(matrix, func) {
        // Apply a function to every element of matrix
        return new Matrix(matrix.rows, matrix.cols).map((e, i, j) => func(matrix.data[i][j], i, j));
    }
    
    print() {
        console.table(this.data);
        return this;
    }
    
    serialize() {
        return JSON.stringify(this);
    }
    
    avg(i) {
        let arr = [];
        let avgAdd = 0
        let avgAns = 0
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                arr.push(this.data[i][j]);
            }
        }
        for (let i = 0; i < arr.length; i++) {
            avgAdd + arr[i]
        }
        avgAns = avgAdd / arr.length
        return avgAns;
    }
    
    static deserialize(data) {
        if (typeof data == 'string') {
            data = JSON.parse(data);
        }
        let matrix = new Matrix(data.rows, data.cols);
        matrix.data = data.data;
        return matrix;
    }
}
if (typeof module !== 'undefined') {
    module.exports = {NeuralNetwork: NeuralNetwork, Matrix: Matrix};
}
