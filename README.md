# svmjs
Andrej Karpathy
July 2012

svmjs is a lightweight implementation of the SMO algorithm to train a binary
Support Vector Machine. As this uses the dual formulation, it also supports
arbitrary kernels. Correctness test, together with MATLAB reference code
are in /test.

## Online GUI demo

Can be found here: http://cs.stanford.edu/~karpathy/svmjs/demo/ 

Corresponding code is inside /demo directory.

## Usage

The simplest use case:
```javascript
// include the library
<script src="./svmjs/lib/svm.js"></script>
<script>
svm = svmjs.SVM();
svm.train(data, labels);
testlabels = svm.predict(testdata);
</script>
```
Here, `data` and `testdata` are a 2D, NxD array of floats, `labels` and `testlabels`
is an array of size N that contains 1 or -1. You can also query for the raw margins:
```javascript
margins = svm.margins(testdata);
margin = svm.marginOne(testadata[0]);
```

The library supports arbitrary kernels, but currently comes with linear and rbf kernel:
```javascript
svm.train(data, labels, { kernel: function(v1,v2){ /* return K(v1, v2) */} }); // arbitrary function
svm.train(data, labels, { kernel: svmjs.linearKernel });
svm.train(data, labels, { kernel: svmjs.makeRbfKernel(0.5) }); // sigma = 0.5
```

For linear kernels, you can also query the weights and offset directly:
```javascript
wb= svm.getWeights(); 
//wb.w is array of weights and wb.b is the bias term
```

For training you can pass in several options. Here are the defaults:
```javascript
var options = {};
/* For C, Higher = you trust your data more. Lower = more regularization. 
Should be in range of around 1e-2 ... 1e5 at most. */
options.C = 1.0; 
options.tol = 1e-4; // do not touch this unless you're pro
options.maxiter = 10000; // if you have a larger problem, you may need to increase this
options.kernel = svmjs.linearKernel; // discussed above
options.numpasses = 10; // increase this for higher precision of the result. (but slower)
svm.train(data, labels, options);
```

## Implementation details
The SMO algorithm is very space efficient, so you need not worry about 
running out of space no matter how large your problem is. However, you do need to
worry about runtime efficiency. In practice, there are many heuristics one can 
use to select the pair of alphas (i,j) to optimize and this uses a rather naive
approach. If you have a large and complex problem, you will need to increase
maxiter a lot. (or don't use Javascript!)

If you intend to use only linear SVM and are worried about efficiency, I recommend 
you train it, get the weights out with getWeights(), and use them directly in 
your code from then on.

## License
MIT
