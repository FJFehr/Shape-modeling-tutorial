object Tutorial_6 extends App {
  /*Building a shape model from data
  Fabio Fehr
  21 May 2020
  * */

  import scalismo.geometry._
  import scalismo.common._
  import scalismo.ui.api._
  import scalismo.mesh._
  import scalismo.io.{StatisticalModelIO, MeshIO}
  import scalismo.statisticalmodel._
  import scalismo.registration._
  import scalismo.statisticalmodel.dataset._
  import scalismo.numerics.PivotedCholesky.RelativeTolerance
  scalismo.initialize()
  implicit val rng = scalismo.utils.Random(42)

  val ui = ScalismoUI()

  val dsGroup = ui.createGroup("datasets")

  val meshFiles = new java.io.File("datasets/nonAlignedFaces/").listFiles

  val (meshes, meshViews) = meshFiles.map(meshFile => {
    val mesh = MeshIO.readMesh(meshFile).get
    val meshView = ui.show(dsGroup, mesh, "mesh")
    (mesh, meshView) // return a tuple of the mesh and the associated view
  }) .unzip // take the tuples apart, to get a sequence of meshes and one of meshViews

  // Notice. The meshes are not aligned, but in correspondence


  // Rigidly aligning the data:
  val reference = meshes.head // first item of data
  val toAlign : IndexedSeq[TriangleMesh[_3D]] = meshes.tail // rest of data

  // grab some points and get their landmarks
  val pointIds = IndexedSeq(2214, 6341, 10008, 14129, 8156, 47775)
  val refLandmarks = pointIds.map{id => Landmark(s"L_$id", reference.pointSet.point(PointId(id))) }

  // This iterates through each mesh, gets the same landmarks as the ref,uses a rigid transformation
  val alignedMeshes = toAlign.map { mesh =>
    val landmarks = pointIds.map{id => Landmark("L_"+id, mesh.pointSet.point(PointId(id)))}
    val rigidTrans = LandmarkRegistration.rigid3DLandmarkRegistration(landmarks, refLandmarks, center = Point(0,0,0))
    mesh.transform(rigidTrans)
  }

  // remove previous visuals
//  (0 to 4).map(i => meshViews(i).remove())

  // Visualise the aligned meshes
//  val meshViewAligned = alignedMeshes.map{mesh =>
//    ui.show(dsGroup, mesh, "alignedMesh")
//  } // It works!

 //  Building a discrete Gaussian process from data

  // Get all deformation vectors for our data
  val defFields = alignedMeshes.map{ m =>
    val deformationVectors = reference.pointSet.pointIds.map{ id : PointId =>
      m.pointSet.point(id) - reference.pointSet.point(id)
    }.toIndexedSeq

    DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]](reference.pointSet, deformationVectors)
  }

  // earning the shape variations from this deformation fields is done by calling the method createUsingPCA
  // of the DiscreteLowRankGaussianProcess class. Note that the deformation fields need to be interpolated,
  // such that we are sure that they are defined on all the points of the reference mesh.

  val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()
  val continuousFields = defFields.map(f => f.interpolate(interpolator) )

  // this mean and variance have used only the data thus this is linear and uses PCA!
  val gp = DiscreteLowRankGaussianProcess.createUsingPCA(reference.pointSet, continuousFields, RelativeTolerance(1e-8))

  //ui.show(dsGroup, gp.mean, "meanMesh") // mean field, you cant make a mesh without the model

  val model = StatisticalMeshModel(reference, gp.interpolate(interpolator))
  val modelGroup = ui.createGroup("model")
  val ssmView = ui.show(modelGroup, model, "model")

  // Here is an easier way to build a model using data collections

  // We can create a DataCollection by providing a reference mesh,
  // and a sequence of meshes, which are in correspondence with this reference.
  val dc = DataCollection.fromMeshSequence(reference, toAlign)._1.get

  val item0 :DataItem[_3D] = dc.dataItems(0)
  val transform : Transformation[_3D] = item0.transformation

  // Easily creates a PDM
  val modelNonAligned = StatisticalMeshModel.createUsingPCA(dc).get


  val modelGroup2 = ui.createGroup("modelGroup2")
  ui.show(modelGroup2, modelNonAligned, "nonAligned")
}
