/**
 * Load and draw *.obj file on canvas, and save as image.
 * *.obj file can be loaded as contour or depth map.
 *
 * @constructor
 * @param divEle {object} div element to draw
 */
var ObjCanvas = function(divEle) {
	this._divEle = divEle

	this._mouseX = 0, this._mouseY = 0;
	this._windowHalfX = window.innerWidth / 2;
	this._windowHalfY = window.innerHeight / 2;


	this._divSizeX = parseInt(divEle.style.width);
	this._divSizeY = parseInt(divEle.style.height);

	// camera
	this._cameraPosZ = 2400;
	this._camera = new THREE.PerspectiveCamera( 45, this._divSizeX / this._divSizeY, 1000, 2800 );
	this._camera.position.z = this._cameraPosZ;

	// control
	this._controls = new THREE.OrbitControls(this._camera);

	// scene
	this._scene = new THREE.Scene();
	var ambient = new THREE.AmbientLight( 0x40f040 );
	this._scene.add( ambient );

	// Manager for loading process
	var manager = new THREE.LoadingManager();
	manager.onProgress = function ( item, loaded, total ) {
		console.log( item, loaded, total );
	};
	var texture = new THREE.Texture();
	this._onProgress = function ( xhr ) {
		if ( xhr.lengthComputable ) {
			var percentComplete = xhr.loaded / xhr.total * 100;
			console.log( Math.round(percentComplete, 2) + '% downloaded' );
		}
	};
	this._onError = function ( xhr ) {
	};

	// Loading texture image
	var loader = new THREE.ImageLoader( manager );
	bLoadTexture = false;
	if (bLoadTexture) {
		loader.load( TEXTURE_PATH, function ( image ) {
			texture.image = image;
			texture.needsUpdate = true;
		} );
	}

	// Settings for renderer and screen
	this._renderer = new THREE.WebGLRenderer({
		preserveDrawingBuffer: true, // in order to save canvas as image file
	});
	this._renderer.setPixelRatio( this._divSizeX / this._divSizeY );
	this._renderer.setSize( this._divSizeX, this._divSizeY );
	this._renderer.setClearColor( new THREE.Color(0xffffff) );

	divEle.appendChild( this._renderer.domElement );

	document.addEventListener( 'mousemove', this.onDocumentMouseMove, false );
};

/**
 * Save what is shown on canvas as image.
 *
 * @param [fileName] {string} File name of saved image. Default value is object name with extension "png".
 */
ObjCanvas.prototype.save = function(fileName) {

	if (!fileName) {
		fileName = this.getObjectName() + ".png";
	}

	var cvs = this._divEle.getElementsByTagName("canvas")[0];

	if (!this._bDepthMap) {
		// save contour as image
		this.addContourTo2DCanvas(cvs, function(canvas2d) {
			canvas2d.toBlob(function ( blob ) {
				saveAs( blob, fileName );
			}, "image/png");
		});
	} else {
		// save depth map as image
		cvs.toBlob(function ( blob ) {
			saveAs( blob, fileName );
		}, "image/png");
	}

};

/**
 * Event handler for mouse move.
 */
ObjCanvas.prototype.onDocumentMouseMove = function ( event ) {
	this._mouseX = ( event.clientX - this._windowHalfX ) / 2;
	this._mouseY = ( event.clientY - this._windowHalfY ) / 2;
};

/**
 * Request to animate and redraw.
 */
ObjCanvas.prototype.animate = function() {
	//requestAnimationFrame( this.animate );
	// http://stackoverflow.com/questions/22039180/failed-to-execute-requestanimationframe-on-window-the-callback-provided-as
	requestAnimationFrame( this.animate.bind(this) );
	this.render();
	this._controls.update();
};

/**
 * Conduct rendering.
 */
ObjCanvas.prototype.render = function () {
	this._scene.position.y = this._scene.position.y + 0.3;
	this._camera.lookAt( this._scene.position );
	this._scene.position.y = this._scene.position.y - 0.3;
	this._renderer.render( this._scene, this._camera );
};


/**
 * load *.obj file.
 *
 * @param filePath {string} file path
 * @param options {object} option settings<br>
 *            options.bDepthMap {boolean} true if you want to draw as depth map.<br>
 *            options.thresholdAngle {number} Threshold angle(degree) for edges to be drawn as contour.
 *                Default value is 15.0.<br>
 */
ObjCanvas.prototype.loadObjFile = function(filePath, options) {
	console.log("start loadObjFile:" + this._objName);

	var _this = this;

	var s = filePath.lastIndexOf("/");
	var e = filePath.lastIndexOf(".");
	this._objName = filePath.substring(s+1, e);

	var bDepthMap = false;
	if (options && options.bDepthMap) {
		bDepthMap = true;
	}
	this._bDepthMap = bDepthMap;

	// Threshold angle to detect contour(degree)
	if (options && options.thresholdAngle) {
		this.thresholdAngle = options.thresholdAngle;
	} else {
		this.thresholdAngle = 15.0;
	}


	// Manager for load processing
	var manager = new THREE.LoadingManager();
	manager.onProgress = function ( item, loaded, total ) {
		console.log( item, loaded, total );
	};

	// model
	// load OBJ file
	var loader = new THREE.OBJLoader( manager );
	loader.load( filePath, function ( object ) {
		var edges = null;
		object.traverse( function ( child ) {
			if (bLoadTexture && child instanceof THREE.Mesh ) {
				child.material.map = texture;
			}
			if (child instanceof THREE.Mesh) {

				if (bDepthMap) {
					var depthMaterial = new THREE.MeshDepthMaterial({
					});
					
					// it is necessary to draw as depth map
					child.material = depthMaterial;
				} else {
					// Draw Contour
					var eGeometry = new THREE.EdgesGeometry( child.geometry, _this.thresholdAngle );
					var eMaterial = new THREE.LineBasicMaterial( { color: 0x0000000, linewidth: 8 } );
					edges = new THREE.LineSegments( eGeometry, eMaterial );
					child.add(edges);
				}

			}
		} );

		// Bounding box
		var bbox = new THREE.BoundingBoxHelper( object, 0xff0000 );
		bbox.update();
		var bmax = bbox.box.max;
		var bmin = bbox.box.min;

		var xyAbsMax = Math.max(Math.max(Math.abs(bmax.x), Math.abs(bmin.x)), Math.max(Math.abs(bmax.y), Math.abs(bmin.y)));

		var zObjMax = Math.max(Math.abs(bmax.z), Math.abs(bmin.z));
		var distObjCamera = xyAbsMax*(1.0/Math.tan(22.5*Math.PI/180));
		_this._camera.position.z =  + zObjMax + distObjCamera * 0.7;
		console.log(_this._camera.position.z);

		var xyzAbsMax = Math.max(Math.max(Math.abs(bmax.z), Math.abs(bmin.z)), xyAbsMax);
		_this._camera.near = _this._camera.position.z * 0.8 -  xyzAbsMax;
		_this._camera.far = bmax.z - bmin.z + 4.5*distObjCamera;
		_this._camera.updateProjectionMatrix();

		_this._scene.add( object );
		
		// test
		_this._camera.position.y = _this._scene.position.y + 0.3;
		_this._camera.position.x = _this._scene.position.x;

		console.log("end add scene:" + _this._objName);

	}, _this._onProgress, _this._onError );

	console.log("end loadObjFile:" + this._objName);
};

/**
 * Load Uint8Array as an Object.<br>
 *
 * @param array {Uint8Array} Array obtained from depth map image.
 *     Size of the array should be width * height of div element
 *     specified by argument of constructor.
 * @deprecated This method is under development.
 */
ObjCanvas.prototype.loadUint8Array = function (array) {
	var _this = this;
	this._objName = "Load from depth data";
	this._bDepthMap = false;

	var width = parseInt(this._divEle.style.width, 10);
	var height = parseInt(this._divEle.style.height, 10);

	console.log(width, height);


	var arMax = -1;
	var arMin = 256;
	for (var i=0;i<array.length;i++) {
		if (array[i]===0) continue;
		arMax = arMax < array[i] ? array[i] : arMax;
		arMin = arMin > array[i] ? array[i] : arMin;
	}
	console.log(arMax);
	console.log(arMin);


	// create geometry from depth array
	var geometry = new THREE.Geometry();

	// vertex data of geometry
	var uvs = [];
	var ins = new Array(width);
	for (var i=0;i<width;i++) {
		ins[i] = new Array(height);
	}
	var outs = new Array(width);
	for (var i=0;i<width;i++) {
		outs[i] = new Array(height);
	}
	var geomIndex = new Array(2);
	geomIndex[0] = ins;
	geomIndex[1] = outs;

	var invWidth = 1.0 / width;
	var invHeight = 1.0 / height;
	var ratioDepthTo255 = 255.0/(arMax-arMin);
	for(var y = 0 ; y < height ; y++) {
		var yMulWidth = y*width;
		for(var x = 0 ; x < width ; x++) {
			if (array[x+yMulWidth] === 0) {
				continue;
			}
			geomIndex[0][x][y] = uvs.length;
			geometry.vertices.push(new THREE.Vector3(x - width*0.5, y - height*0.5, 0.5*(array[x + yMulWidth] - arMin) * ratioDepthTo255 ));
			uvs.push(new THREE.Vector2(x * invWidth, y * invHeight));
		}
	}

	// face data of geometry
	for(var y = 0 ; y < height - 1 ; y++) {
		for(var x = 0 ; x < width - 1 ; x++) {
			var i1 = geomIndex[0][x][y];
			var i2 = geomIndex[0][x+1][y];
			var i3 = geomIndex[0][x+1][y+1];
			var i4 = geomIndex[0][x][y+1];
		
			// triangle i1 i2 i4
			if (i1 != undefined && i2 != undefined  && i4 != undefined ) {
				geometry.faces.push(
					new THREE.Face3(i1, i2, i4)
				);
				geometry.faceVertexUvs[0].push(
					[uvs[i1], uvs[i2], uvs[i4]]
				);
			}
			// triangle i2 i3 i4
			if (i2 != undefined && i3 != undefined && i4 != undefined) {
				//var b = x + y * width;
				geometry.faces.push(
					new THREE.Face3(i2, i3, i4)
				);
				geometry.faceVertexUvs[0].push(
					[uvs[i2], uvs[i3], uvs[i4]]
				);
			}
			
			/*
			var b = x + y * width;
			var m = (x + y) & 1;
			geometry.faces.push(
				new THREE.Face3(b + 0, b +  1, b + 65, null, null, m),
				new THREE.Face3(b + 1, b + 66, b + 65, null, null, m)
			);
			geometry.faceVertexUvs[0].push(
				[uvs[b + 0], uvs[b +  1], uvs[b + 65]],
				[uvs[b + 1], uvs[b + 66], uvs[b + 65]]
			);
			*/
		}
	}

	// hidden planes
	for(var y = 0 ; y < height ; y++) {
		var yMulWidth = y*width;
		for(var x = 0 ; x < width ; x++) {
			if (array[x+yMulWidth] === 0) {
				continue;
			}
			geomIndex[1][x][y] = uvs.length;
			geometry.vertices.push(new THREE.Vector3(x - width*0.5, y - height*0.5, - 0.5 * (array[x + yMulWidth] - arMin) * ratioDepthTo255 ));
			uvs.push(new THREE.Vector2(x * invWidth, y * invHeight));
		}
	}

	// face data of geometry
	for(var y = 0 ; y < height - 1 ; y++) {
		for(var x = 0 ; x < width - 1 ; x++) {
			var i1 = geomIndex[1][x][y];
			var i2 = geomIndex[1][x+1][y];
			var i3 = geomIndex[1][x+1][y+1];
			var i4 = geomIndex[1][x][y+1];
		
			// triangle i1 i2 i4
			if (i1 != undefined && i2 != undefined  && i4 != undefined ) {
				geometry.faces.push(
					new THREE.Face3(i1, i4, i2)
				);
				geometry.faceVertexUvs[0].push(
					[uvs[i1], uvs[i4], uvs[i2]]
				);
			}
			// triangle i2 i3 i4
			if (i2 != undefined && i3 != undefined && i4 != undefined) {
				//var b = x + y * width;
				geometry.faces.push(
					new THREE.Face3(i2, i4, i3)
				);
				geometry.faceVertexUvs[0].push(
					[uvs[i2], uvs[i4], uvs[i3]]
				);
			}
		}
	}
	
	// 表面と裏面の間
	// TODO:法線の方向を考えるのが面倒なので両面入れている
	for(var y = 1 ; y < height - 1 ; y++) {
		for(var x = 1 ; x < width - 1 ; x++) {
			if (geomIndex[0][x][y] && (geomIndex[0][x-1][y]==undefined || geomIndex[0][x][y-1]==undefined || geomIndex[0][x+1][y]==undefined || geomIndex[0][x][y+1]==undefined)) {
				// 横・縦でつながるならそちらを優先
				var ix, iy;
				if (geomIndex[0][x-1][y]) {
					ix = x-1; iy = y;
				} else if (geomIndex[0][x][y-1]) {
					ix = x; iy = y-1;
				} else if (geomIndex[0][x+1][y]) {
					ix = x+1; iy = y;
				} else if (geomIndex[0][x][y+1]) {
					ix = x; iy = y+1;
				}
				var i0 = geomIndex[0][x][y];
				var i1 = geomIndex[1][x][y];
				var i2 = geomIndex[0][ix][iy];
				var i3 = geomIndex[1][ix][iy];
				geometry.faces.push(
					new THREE.Face3(i0, i1, i2),
					new THREE.Face3(i1, i3, i2),
					new THREE.Face3(i0, i2, i1),
					new THREE.Face3(i1, i2, i3)
				);
				geometry.faceVertexUvs[0].push(
					[uvs[i0], uvs[i1], uvs[i2]],
					[uvs[i1], uvs[i3], uvs[i2]],
					[uvs[i0], uvs[i2], uvs[i1]],
					[uvs[i1], uvs[i2], uvs[i3]]
				);
			}
			if (geomIndex[0][x][y] && (geomIndex[0][x-1][y-1]==undefined || geomIndex[0][x+1][y-1]==undefined || geomIndex[0][x-1][y+1]==undefined || geomIndex[0][x+1][y+1]==undefined)) {
				// 横・縦でつながるならそちらを優先
				var ix, iy;
				if (geomIndex[0][x-1][y-1]) {
					ix = x-1; iy = y-1;
				} else if (geomIndex[0][x+1][y-1]) {
					ix = x+1; iy = y-1;
				} else if (geomIndex[0][x-1][y+1]) {
					ix = x-1; iy = y+1;
				} else if (geomIndex[0][x+1][y+1]) {
					ix = x+1; iy = y+1;
				}
				var i0 = geomIndex[0][x][y];
				var i1 = geomIndex[1][x][y];
				var i2 = geomIndex[0][ix][iy];
				var i3 = geomIndex[1][ix][iy];
				geometry.faces.push(
					new THREE.Face3(i0, i1, i2),
					new THREE.Face3(i1, i3, i2),
					new THREE.Face3(i0, i2, i1),
					new THREE.Face3(i1, i2, i3)
				);
				geometry.faceVertexUvs[0].push(
					[uvs[i0], uvs[i1], uvs[i2]],
					[uvs[i1], uvs[i3], uvs[i2]],
					[uvs[i0], uvs[i2], uvs[i1]],
					[uvs[i1], uvs[i2], uvs[i3]]
				);
			}

		}
	}
	
/*
	// args:width, height, nPlanesX, nPlanesY
	var geometry = new THREE.PlaneBufferGeometry(width, height, width-1, height-1);
	//geometry.rotateX( - Math.PI / 2 );
	var vertices = geometry.attributes.position.array;
	for (var x=0;x<width+1;x++) {
		for (var y=0;y<height+1;y++) {
			var index = (y*(width+1)+x);
			vertices[index * 3 + 2] = array[index];
		}
	}
*/


	// TODO:want to make it smooth
	// http://stackoverflow.com/questions/13880497/add-subdivision-to-a-geometry
	// http://stackoverflow.com/questions/12994175/creating-a-cube-with-rounded-corners-in-three-js
	//var modifier = new THREE.SubdivisionModifier(2);
	//modifier.modify( geometry );

	geometry.rotateZ(Math.PI);

	geometry.computeFaceNormals();
	geometry.computeVertexNormals();

	var mesh = new THREE.Mesh( geometry, new THREE.MeshPhongMaterial() );
//	var mesh = new THREE.Mesh( geometry, new THREE.MeshBasicMaterial() );
//	var mesh = new THREE.Mesh( geometry, new THREE.MeshDepthMaterial() );
//	console.log(geometry);

/*
	// Bounding box
	var bbox = new THREE.BoundingBoxHelper( mesh, 0xff0000 );
	bbox.update();
	var bmax = bbox.box.max;
	var bmin = bbox.box.min;

	var xyAbsMax = Math.max(Math.max(Math.abs(bmax.x), Math.abs(bmin.x)), Math.max(Math.abs(bmax.y), Math.abs(bmin.y)));

	//_this._camera.position.z = bmax.z - bmin.z + 2*xyAbsMax;
	var zObjMax = Math.max(Math.abs(bmax.z), Math.abs(bmin.z));
	var distObjCamera = xyAbsMax*(1.0/Math.tan(22.5*Math.PI/180));
	_this._camera.position.z =  + zObjMax + distObjCamera;
	var xyzAbsMax = Math.max(Math.max(Math.abs(bmax.z), Math.abs(bmin.z)), xyAbsMax);
	_this._camera.near = 0;//_this._camera.position.z * 0.8 -  xyzAbsMax;//     0.7*distObjCamera;
	_this._camera.far = bmax.z - bmin.z + 4.5*distObjCamera;
	_this._camera.updateProjectionMatrix();
	
	console.log(_this._camera.position.z);
	console.log(_this._camera.near);
	console.log(_this._camera.far);
	console.log(distObjCamera);
	console.log(xyAbsMax);
*/
	_this._scene.add( mesh );

	// last from here
	var directionalLight = new THREE.DirectionalLight( 0x20ff20 );
	directionalLight.position.set( -1000, 1000, 0 );
	_this._scene.add( directionalLight );


	_this._camera.position.z = 1000;
	_this._camera.near = 0.1;
	_this._camera.far = 7200;
	_this._camera.updateProjectionMatrix();
};

/*
 * Add contours to objects drawn on canvas(webgl).
 * Canvas with contour is passed to the callback as argument.
 *
 * @param canvas {canvas} Canvas contours are added.
 * @param callback {function} Callback function to use canvas with contour.
 *     This function takes one argument(canvas).
 * @private
 */
ObjCanvas.prototype.addContourTo2DCanvas = function (canvas, callback) {
	var canvas2d = document.createElement("canvas");
	canvas2d.width = this._divSizeX;
	canvas2d.height = this._divSizeY;
	var context2d = canvas2d.getContext('2d');
	var imgObj = new Image();
	imgObj.onload = function () {
		context2d.drawImage(imgObj, 0, 0);

		// Get pixel data from canvas
		var imageData = context2d.getImageData(0, 0, canvas.width, canvas.height);
		var width = imageData.width, height = imageData.height;
		var pixels = imageData.data;  // 1 pixel consists of 4(RGBA) elements of pixel array

		var drawContourPixel = function(base, baseNext) {
			if (pixels[base + 0]==255 && pixels[base + 1]==255 && pixels[base + 2]==255) {
				if (pixels[baseNext + 0]!=255 && pixels[baseNext + 1]!=255 && pixels[baseNext + 2]!=255) {
					pixels[baseNext + 0] = 0; // R
					pixels[baseNext + 1] = 0; // G
					pixels[baseNext + 2] = 0; // B
				}
			}
		};

		// Add contour by checking if each pixel needs to be overwritten
		for (var y = 0; y < height; ++y) {
			// From left to right
			for (var x = 0; x < width-1; ++x) {
				var base = (y * width + x) * 4;
				var baseNext = (y * width + x+1) * 4;
				drawContourPixel(base, baseNext);
			}
			// From right to left
			for (var x = width-1; x > 0; x--) {
				var base = (y * width + x) * 4;
				var baseNext = (y * width + x-1) * 4;
				drawContourPixel(base, baseNext);
			}
		}
		for (var x = 0; x < width; ++x) {
			// From bottom to top
			for (var y = 0; y < height-1; ++y) {
				var base = (y * width + x) * 4;
				var baseNext = ((y+1) * width + x) * 4;
				drawContourPixel(base, baseNext);
			}
			// From top to bottom
			for (var y = height-1; y > 0; y--) {
				var base = (y * width + x) * 4;
				var baseNext = ((y-1) * width + x) * 4;
				drawContourPixel(base, baseNext);
			}
		}
		// All pixels other than coutour should be drawn as while
		for (var y = 0; y < height; ++y) {
			for (var x = 0; x < width; ++x) {
				var base = (y * width + x) * 4;
				if (pixels[base + 0]!=0 && pixels[base + 1]!=0 && pixels[base + 2]!=0) {
					pixels[base + 0] = 255; // R
					pixels[base + 1] = 255; // G
					pixels[base + 2] = 255; // B
				}
			}
		}

		// Draw imageData overwritten above on canvas
		context2d.putImageData(imageData, 0, 0);
		
		callback(canvas2d);
	};
	imgObj.src = canvas.toDataURL("image/png");
};


/**
 * Save multiple images seen from different 26 angles automatically.
 *
 * @param [distRatio] {number} Ratio multiplied to distance from camera to object.
 *     The default value is 1.0.
 */
ObjCanvas.prototype.autoSave = function(distRatio) {
	console.log("start autoSave:" + this._objName);

	// Save camera values to restore at the end of this method
	var prevX = this._camera.position.x;
	var prevY = this._camera.position.y;
	var prevZ = this._camera.position.z;

	if (!distRatio) {
		distRatio = 1.0;
	}
	var fileNamePostfix = "";
	if (distRatio !== 1.0) {
		fileNamePostfix = "_" + distRatio;
	}

	var dist = this._camera.position.z * distRatio;
	console.log(this._objName + " " + distRatio);
	
	var count = 0;
	for (var angle=0;angle<360;angle+=45) {
		this._camera.position.x = dist * Math.sin(angle*(Math.PI/180));
		this._camera.position.y = 0;
		this._camera.position.z = dist * Math.cos(angle*(Math.PI/180));
		this.render();
		
		this.save(this._objName + fileNamePostfix + "_" + (this._bDepthMap ? "d" : "s") + ("00"+count).slice(-2) + ".png");
		count+=1;
	}

	for (var angle=0;angle<360;angle+=45) {
		this._camera.position.x = (dist * Math.sin(45*(Math.PI/180))) * Math.sin(angle*(Math.PI/180));
		this._camera.position.y = (dist * Math.sin(45*(Math.PI/180)));
		this._camera.position.z = (dist * Math.sin(45*(Math.PI/180))) * Math.cos(angle*(Math.PI/180));
		this.render();
		
		this.save(this._objName + fileNamePostfix + "_" + (this._bDepthMap ? "d" : "s") + ("00"+count).slice(-2) + ".png");
		count+=1;
	}
	this._camera.position.x = 0;
	this._camera.position.y = dist;
	this._camera.position.z = 0;
	this.render();
	this.save(this._objName + fileNamePostfix + "_" + (this._bDepthMap ? "d" : "s") + ("00"+count).slice(-2) + ".png");
	count+=1;

	for (var angle=0;angle<360;angle+=45) {
		this._camera.position.x = (dist * Math.sin(45*(Math.PI/180))) * Math.sin(angle*(Math.PI/180));
		this._camera.position.y = -(dist * Math.sin(45*(Math.PI/180)));
		this._camera.position.z = (dist * Math.sin(45*(Math.PI/180))) * Math.cos(angle*(Math.PI/180));
		this.render();
		
		this.save(this._objName + fileNamePostfix + "_" + (this._bDepthMap ? "d" : "s") + ("00"+count).slice(-2) + ".png");
		count+=1;
	}
	this._camera.position.x = 0;
	this._camera.position.y = -dist;
	this._camera.position.z = 0;
	this.render();
	this.save(this._objName + fileNamePostfix + "_" + (this._bDepthMap ? "d" : "s") + ("00"+count).slice(-2) + ".png");
	count+=1;

	this._camera.position.x = prevX;
	this._camera.position.y = prevY;
	this._camera.position.z = prevZ;

	console.log("end autoSave:" + this._objName);
};

/**
 * Returns name of object.<br>
 * When object was loaded from *.obj file, this function will return file name of it.
 *
 * @return {string} Name of object
 */
ObjCanvas.prototype.getObjectName = function() {
	return this._objName;
};
