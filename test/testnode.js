/* Run with `node testnode.js` */
var assert = require("assert"),
    svm = require("../lib/svm");

var SVM = new svm.SVM();

var options = {
  kernel: svm.makeRbfKernel(0.5)
}

SVM.train([[0,0], [0,1], [1,0], [1,1]], [-1, 1, 1, -1], options);

assert.equal(SVM.predict([[0,0]]), -1);
assert.equal(SVM.predict([[0,1]]), 1);
assert.equal(SVM.predict([[1,0]]), 1);
assert.equal(SVM.predict([[1,1]]), -1);

var json = SVM.toJSON();

var SVM2 = new svm.SVM();

assert.equal(SVM2.predict([[0,1]]), -1);

SVM2.fromJSON(json, svm.makeRbfKernel(0.5));

assert.equal(SVM2.predict([[0,0]]), -1);
assert.equal(SVM2.predict([[0,1]]), 1);
assert.equal(SVM2.predict([[1,0]]), 1);
assert.equal(SVM2.predict([[1,1]]), -1);
