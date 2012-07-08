function Vector2D(xinit, yinit)
{
  this.x = xinit;
  this.y = yinit;

  this.set = function(x, y)
  {
    this.x = x;
    this.y = y;
  }

  this.copy = function(v)
  {
    this.x = v.x;
    this.y = v.y;
  }

  this.lengthSquared = function()
  {
    return this.x * this.x + this.y * this.y;
  }

  this.length = function()
  {
    return Math.sqrt(this.lengthSquared());
  }

  this.scaleEq = function(s)
  {
    this.x *= s;
    this.y *= s;
  }

  this.minusEq = function(v)
  {
    this.x -= v.x;
    this.y -= v.y;
  }

  this.plusEq = function(v)
  {
    this.x += v.x;
    this.y += v.y;
  }

  this.convolveEq = function(v)
  {
    this.x *= v.x;
    this.y *= v.y;
  }

  this.dot = function(v)
  {
    return(this.x * v.x + this.y * v.y);
  }

  this.normalizeEq = function()
  {
    this.scaleEq(1/this.length());
  }
}

