import 'source-map-support/register';

import { APIGatewayProxyEvent, APIGatewayProxyResult, Callback, Context } from 'aws-lambda';
import sizeOf from 'image-size';
import axios from 'axios';
import fs from 'fs';
import { Readable } from 'stream';
import loadTf from 'tfjs-node-lambda';

export const handler = (
  event: APIGatewayProxyEvent,
  context: Context,
  callback: Callback
): APIGatewayProxyResult => {
  const response = {
    statusCode: 200,
    body: 'ok',
  };

  const runModel = async () => {
    function fetchInputImage() {
      return tfnode.tidy(() => {
        const { height, width } = calculateAspectRatioFit(
          Math.round(imgDims[1]),
          Math.round(imgDims[0]),
          320,
          240
        );
        const imageDimsReducedTo: [number, number] = [height, width];
        const decodedImg = tfnode.node.decodeImage(imageBuffer, 3);

        const batchedImage = decodedImg.toFloat();

        const resizedImage = tfnode.image.resizeBilinear(batchedImage, imageDimsReducedTo, true);

        const clippedImage = tfnode.clipByValue(resizedImage, 0.0, 255.0);

        return clippedImage;
      });
    }
    function predictSaliency() {
      return tfnode.tidy(() => {
        const modelOutput = model.predict(fetchInputImage()) as any;
        const resizedOutput = tfnode.image.resizeBilinear(modelOutput, imgDims, true);
        const clippedOutput = tfnode.clipByValue(resizedOutput, 0.0, 255.0);
        return clippedOutput.squeeze();
      });
    }

    const url2 =
      'https://github.com/jlarmstrongiv/tfjs-node-lambda/releases/download/v2.0.11/nodejs14.x-tf2.8.6.br';

    const axiosresponse = await axios.get(url2, { responseType: 'arraybuffer' });

    const readStream = Readable.from(axiosresponse.data);

    const tfnode: typeof import('@tensorflow/tfjs-node') = await loadTf(readStream);

    const { arrayBuffer } = await fetch('https://dummyimage.com/600x400/000/fff');
    const imageBuffer = await streamToString(arrayBuffer);
    const dimensions = sizeOf(imageBuffer);

    const imgDims: [number, number] = [dimensions.height, dimensions.width];

    const modelURL = 'https://storage.googleapis.com/msi-net/model/very_high/model.json';
    const model = await tfnode.loadGraphModel(modelURL);
    // tfnode.tidy(() => model.predict(fetchInputImage())); // warmup
    const saliencyMap = predictSaliency();

    const img = await tfnode.node.encodeJpeg(
      saliencyMap
        .mul(255)
        .cast('int32')
        .reshape([...imgDims, 1])
    );

    fs.writeFileSync('./op/out.jpg', img);

    saliencyMap.dispose();
    model.dispose();
  };

  runModel();

  console.info(`response from:  statusCode: ${response.statusCode} body: ${response.body}`);
  callback(null, response);
  return;
};
const streamToString = (stream): Promise<Buffer> =>
  new Promise((resolve, reject) => {
    const chunks: Uint8Array[] = [];
    stream.on('data', (chunk) => chunks.push(chunk));
    stream.on('error', reject);
    stream.on('end', () => resolve(Buffer.concat(chunks)));
  });

function calculateAspectRatioFit(
  srcWidth: number,
  srcHeight: number,
  maxWidth: number,
  maxHeight: number
) {
  var ratio = Math.min(maxWidth / srcWidth, maxHeight / srcHeight);
  return { width: srcWidth * ratio, height: srcHeight * ratio };
}
