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
data = [[0,0], [0,1], [1,0], [1,1]];
labels = [-1, 1, 1, -1];
svm = new svmjs.SVM();
svm.train(data, labels, {C: 1.0}); // C is a parameter to SVM
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
svm.train(data, labels, { kernel: 'linear' });
svm.train(data, labels, { kernel: 'rbf', rbfsigma: 0.5 }); // sigma in the gaussian kernel = 0.5
```

For training you can pass in several options. Here are the defaults:
```javascript
var options = {};
/* For C, Higher = you trust your data more. Lower = more regularization.
Should be in range of around 1e-2 ... 1e5 at most. */
options.C = 1.0;
options.tol = 1e-4; // do not touch this unless you're pro
options.alphatol = 1e-7; // used for pruning non-support vectors. do not touch unless you're pro
options.maxiter = 10000; // if you have a larger problem, you may need to increase this
options.kernel = svmjs.linearKernel; // discussed above
options.numpasses = 10; // increase this for higher precision of the result. (but slower)
svm.train(data, labels, options);
```

Rules of thumb: You almost always want to try the linear SVM first and see how that works. You want
to play around with different values of C from about 1e-2 to 1e5, as every dataset is different. `C=1`
is usually a fairly reasonable value. Roughly, C is the cost to the SVM when it mis-classifies one of your
training examples. If you increase it, the SVM will try very hard to fit all your data, which may be good
if you strongly trust your data. In practice, you usually don't want it too high though. If linear kernel 
doesn't work very well, try the rbf kernel. You will have to try different values of both C and just as crucially the sigma for the gaussian kernel. 

The linear SVM should be much faster than SVM with any other kernel. If you want it even faster 
but less accurate, you want to play around with options.tol (try increase a bit). You can also try to
decrease options.maxiter and especially options.numpasses (decrease a bit). 
If you use non-linear svm, you can also speed up the svm at test by playing around with 
options.alphatol (try increase a bit).

If you use linear or rbf kernel (instead of some custom one) you can load and save the svm:
```javascript
var svm = new svmjs.SVM();
var json = svm.toJSON();
var svm2 = new svmjs.SVM();
svm2.fromJSON(json);
```

## Using in node
To use this library in [node.js](http://nodejs.org/), install with `npm`:

```
npm install svm
```

And use like so:

```javascript
var svm = require("svm");
var SVM = new svm.SVM();
SVM.train(data, labels);
```

## Implementation details
The SMO algorithm is very space efficient, so you need not worry about
running out of space no matter how large your problem is. However, you do need to
worry about runtime efficiency. In practice, there are many heuristics one can
use to select the pair of alphas (i,j) to optimize and this uses a rather naive
approach. If you have a large and complex problem, you will need to increase
maxiter a lot. (or don't use Javascript!)

## License
MIT
