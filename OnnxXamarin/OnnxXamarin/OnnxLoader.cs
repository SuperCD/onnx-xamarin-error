using System.IO;
using Microsoft.ML.OnnxRuntime;

namespace OnnxXamarin
{
	public static class OnnxLoader
	{
        private static byte[] LoadModelFromEmbeddedResources(System.Reflection.Assembly assembly, string resourceName)
        {
            using var modelStream = assembly.GetManifestResourceStream(resourceName);
            using var modelMemoryStream = new MemoryStream();
            modelStream.CopyTo(modelMemoryStream);
            return modelMemoryStream.ToArray();
        }

        public static void LoadModel()
        {
            string resourceName = "mobilenetv2-7.onnx";
            //string resouceName = "counter_detector.onnx";
            var modelBytes = LoadModelFromEmbeddedResources(typeof(OnnxLoader).Assembly, resourceName);

            new InferenceSession(modelBytes);
        }
	}
}

