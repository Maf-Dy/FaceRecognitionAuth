import 'dart:io';
import 'dart:isolate';
import 'dart:math';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:face_net_authentication/pages/db/databse_helper.dart';
import 'package:face_net_authentication/pages/models/user.model.dart';
import 'package:face_net_authentication/services/image_converter.dart';
import 'package:flutter/foundation.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as imglib;

class MLService {
  Interpreter? interpreter;
  double threshold = 0.75;
  var assetName = 'facenet.tflite';

  List _predictedData = [];
  List get predictedData => _predictedData;

  Future initialize() async {
    late Delegate delegate;
    try {
      if (Platform.isAndroid) {
        delegate = GpuDelegateV2(
          options: GpuDelegateOptionsV2(
            isPrecisionLossAllowed: false,
            inferencePreference: TfLiteGpuInferenceUsage.fastSingleAnswer,
            inferencePriority1: TfLiteGpuInferencePriority.minLatency,
            inferencePriority2: TfLiteGpuInferencePriority.auto,
            inferencePriority3: TfLiteGpuInferencePriority.auto,
          ),
        );
      } else if (Platform.isIOS) {
        delegate = GpuDelegate(
          options: GpuDelegateOptions(
              allowPrecisionLoss: true,
              waitType: TFLGpuDelegateWaitType.active),
        );
      }
      var interpreterOptions = InterpreterOptions()..addDelegate(delegate);
      interpreterOptions.threads = 2;


      interpreter = await Interpreter.fromAsset(assetName,
          options: interpreterOptions);
    } catch (e) {
      print('Failed to load model.');
      print(e);
    }
  }


   Isolate? _isolate;
   late ReceivePort receivePort;

  // Call this to kill isolate
  void _stopIsolate() {
    if (_isolate != null) {
      receivePort.close();
      _isolate?.kill(priority: Isolate.immediate);
      _isolate = null;
    }
  }

  void _onDataCallback(dynamic input) {
    //print(messageFromIsolate);

    input = (input as List).reshape([1, 112, 112, 3]);
    List output = List.generate(1, (index) => List.filled(512, 0));

    interpreter?.run(input, output);
    output = output.reshape([512]);

    _predictedData = List.from(output);

  }

  static void _entryPoint(IsolateData isolateData) async {
    List input = _preProcess(isolateData.cameraImage, isolateData.face);
    isolateData.sendPort.send(input);
  }


  void setCurrentPrediction(CameraImage cameraImage, Face? face) async {
    if (interpreter == null) throw Exception('Interpreter is null');
    if (face == null) throw Exception('Face is null');

    receivePort = ReceivePort();

    IsolateData isolateData = IsolateData(
        cameraImage, face, receivePort.sendPort);

    _isolate = await Isolate.spawn(
      _entryPoint,
      isolateData,
    );

    receivePort.listen(_onDataCallback, onDone: () {
      _stopIsolate();
    });

  }

  Future<User?> predict() async {
    return _searchResult(_predictedData);
  }

  static List _preProcess(CameraImage image, Face faceDetected) {
    imglib.Image croppedImage = _cropFace(image, faceDetected);
    imglib.Image img = imglib.copyResizeCropSquare(croppedImage, 112);

    Float32List imageAsList = imageToByteListFloat32(img);
    return imageAsList;
  }

  static imglib.Image _cropFace(CameraImage image, Face faceDetected) {
    imglib.Image convertedImage = _convertCameraImage(image);
    double x = faceDetected.boundingBox.left - 10.0;
    double y = faceDetected.boundingBox.top - 10.0;
    double w = faceDetected.boundingBox.width + 10.0;
    double h = faceDetected.boundingBox.height + 10.0;
    return imglib.copyCrop(
        convertedImage, x.round(), y.round(), w.round(), h.round());
  }

  static imglib.Image _convertCameraImage(CameraImage image) {
    var img = convertToImage(image);
    var img1 = imglib.copyRotate(img, -90);
    return img1;
  }

  static Float32List imageToByteListFloat32(imglib.Image image) {
    var convertedBytes = Float32List(1 * 112 * 112 * 3);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;

    for (var i = 0; i < 112; i++) {
      for (var j = 0; j < 112; j++) {
        var pixel = image.getPixel(j, i);
        buffer[pixelIndex++] = (imglib.getRed(pixel) - 128) / 128;
        buffer[pixelIndex++] = (imglib.getGreen(pixel) - 128) / 128;
        buffer[pixelIndex++] = (imglib.getBlue(pixel) - 128) / 128;
      }
    }
    return convertedBytes.buffer.asFloat32List();
  }

  Future<User?> _searchResult(List predictedData) async {
    DatabaseHelper _dbHelper = DatabaseHelper.instance;

    List<User> users = await _dbHelper.queryAllUsers();
    double minDist = 999;
    double currDist = 0.0;
    User? predictedResult;

    for (User u in users) {
      currDist = _euclideanDistance(u.modelData, predictedData);
      print('user: ${u.user} , distance: $currDist');
      if (currDist <= threshold && currDist < minDist) {
        minDist = currDist;
        predictedResult = u;
      }
    }
    return predictedResult;
  }

  double _euclideanDistance(List? e1, List? e2) {
    if (e1 == null || e2 == null) throw Exception("Null argument");

    double sum = 0.0;
    for (int i = 0; i < e1.length; i++) {
      sum += pow((e1[i] - e2[i]), 2);
    }
    return sqrt(sum);
  }

  void setPredictedData(value) {
    this._predictedData = value;
  }

  dispose() {}
}

class IsolateData {
  IsolateData(this.cameraImage, this.face,  this.sendPort);
  CameraImage cameraImage;
  Face face;
  SendPort sendPort;
}
