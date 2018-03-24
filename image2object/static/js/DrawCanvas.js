/**
 * Canvas users can do painting.<br>
 * Drawn image can be obtained as Base64 string.
 *
 * @constructor
 * @param canvasEle {object} canvas element to draw
 * @param [undoStackSize] {number} Size of stack used for undo function.
 *     Negative value indicates that stack size is theoretically inifinite,
 *     and default value is -1. When stack size exceeds this value,
 *     oldest data in the stack is abandoned. Note that each elements in stack
 *     holds copied whole canvas data, so be careful about data size on memory.
 */
var DrawCanvas = function(canvasEle, undoStackSize) {
	var _this = this;
	this._canvasEle = canvasEle;

	if (undoStackSize === undefined || undoStackSize === null) {
		undoStackSize = -1;
	}
	this._undoStackSize = undoStackSize;
	this._undoStack = [];
	// index for undo stack used for next undo call
	this._undoPointer = 0;

	var drawContext = canvasEle.getContext("2d");
	this._drawContext = drawContext;

	// store initial canvas for undo function
	var storedImageData = this._drawContext.getImageData(0, 0, this._canvasEle.width, this._canvasEle.height);
	this._undoStack.push(storedImageData);

	// drawing style
	drawContext.lineCap = "round";
	drawContext.lineWidth = 1;
	drawContext.strokeStyle = 'rgb(0, 0, 0)';
	drawContext.fillStyle = 'rgb(0, 0, 0)';

	var mouseX = null;
	var mouseY = null;
	
	this._isMouseDown = false;

	// On move of mouse pointer
	var onMove = function(e) {
		if (e.buttons === 1 || e.which === 1) {
			var rect = e.target.getBoundingClientRect();
			var X = ~~(e.clientX - rect.left);
			var Y = ~~(e.clientY - rect.top);
			draw(X, Y);
		}
	};
	this._canvasEle.addEventListener('mousemove', onMove, false);

	// On left click by mouse
	var draw = function(X, Y) {
		drawContext.beginPath();
		if (!mouseX) {
			// start drawing from current mouse cursor position
			drawContext.moveTo(X, Y);
		} else {
			// start drawing from previous mouse cursor position
			drawContext.moveTo(mouseX, mouseY);
		}
		drawContext.lineTo(X, Y);
		drawContext.stroke();

		// for next call
		mouseX = X;
		mouseY = Y;
	};
	var onClick = function(e) {
		if (e.button != 0) {
			return;
		}
		
		this._isMouseDown = true;

		var rect = e.target.getBoundingClientRect();
		var X = ~~(e.clientX - rect.left);
		var Y = ~~(e.clientY - rect.top);
		draw(X, Y);
	};
	this._canvasEle.addEventListener('mousedown', onClick, false);

	var drawEnd = function() {
		mouseX = null;
		mouseY = null;
		
		if (this._isMouseDown) {
			// delete no longer used undo data when user draws after calling undo
			while (_this._undoStack.length > _this._undoPointer + 2) {
				_this._undoStack.pop();
			}
		
			// store data for undo function
			var storedImageData = _this._drawContext.getImageData(0, 0, _this._canvasEle.width, _this._canvasEle.height);
			_this._undoStack.push(storedImageData);
			_this._undoPointer++;
		}

		this._isMouseDown = false;
	};
	this._canvasEle.addEventListener('mouseup', drawEnd, false);
	this._canvasEle.addEventListener('mouseout', drawEnd, false);
};

/**
 * Set pen size.
 *
 * @param size {number} Pen size to set.
 * @memberOf DrawCanvas
 */
DrawCanvas.prototype.setPenSize = function (size) {
	this._drawContext.lineWidth = size;
};

/**
 * Set pen color.
 *
 * @param r {number} Red color to set.
 *     The value should be an integer from 0 to 255.
 * @param g {number} Green color to set.
 *     The value should be an integer from 0 to 255.
 * @param b {number} Blue color to set.
 *     The value should be an integer from 0 to 255.
 * @memberOf DrawCanvas
 */
DrawCanvas.prototype.setPenColor = function (r, g, b) {
	this._drawContext.strokeStyle = 'rgb(' + r + ', ' + g + ', ' +  b + ')';
	this._drawContext.fillStyle =  'rgb(' + r + ', ' + g + ', ' +  b + ')';
};

/**
 * Undo the stroke by user.
 *
 * @memberOf DrawCanvas
 */
DrawCanvas.prototype.undo = function () {
	if (this._undoPointer < 1) {
		return;
	}
	this._drawContext.putImageData(this._undoStack[this._undoPointer - 1], 0, 0);
	if (this._undoPointer >= 1) this._undoPointer--;
};

/**
 * Redo the stroke by user.
 *
 * @memberOf DrawCanvas
 */
DrawCanvas.prototype.redo = function () {
	if (this._undoPointer+1 >= this._undoStack.length) {
		return;
	}
	this._drawContext.putImageData(this._undoStack[this._undoPointer + 1], 0, 0);
	if (this._undoPointer < this._undoStack.length) this._undoPointer++;
};

/**
 * Clear what is drawn on canvas.
 *
 * @memberOf DrawCanvas
 */
DrawCanvas.prototype.clear = function () {
	this._drawContext.beginPath();
	this._drawContext.fillStyle = "#ffffff";
	this._drawContext.fillRect(0, 0, this._canvasEle.width, this._canvasEle.height);
};

/**
 * Return Base64 string of what is drawn on canvas.
 *
 * @param [type] {string} Type of base64 encode.
 *     Default value is "image/png".
 * @return {string} Base64 string
 * @memberOf DrawCanvas
 */
DrawCanvas.prototype.toDataURL = function(type) {
	type = type ? type : "image/png";

	return this._canvasEle.toDataURL(type);
};


