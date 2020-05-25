object Tutorial_11 extends App {

  /*Model fitting with Iterative Closest Points
  ICP to find the best rigid transformation between two meshes.
  * */

  import scalismo.geometry._
  import scalismo.common._
  import scalismo.ui.api._
  import scalismo.mesh._
  import scalismo.statisticalmodel.MultivariateNormalDistribution
  import scalismo.numerics.UniformMeshSampler3D
  import scalismo.io.{MeshIO, StatisticalModelIO, LandmarkIO}
  import breeze.linalg.{DenseMatrix, DenseVector}

  scalismo.initialize()
  implicit val rng = scalismo.utils.Random(42)

  val ui = ScalismoUI()

  // Setup. Target and a model

  val targetMesh = MeshIO.readMesh(new java.io.File("datasets/target.ply")).get
  val model = StatisticalModelIO.readStatisticalMeshModel(new java.io.File("datasets/bfm.h5")).get

  val targetGroup = ui.createGroup("targetGroup")
  val targetMeshView = ui.show(targetGroup, targetMesh, "targetMesh")

  val modelGroup = ui.createGroup("modelGroup")
  val modelView = ui.show(modelGroup, model, "model")

  // 1) Find candidate correspondences between the mesh to be aligned and the target one,
  //    by attributing the closest point on the target mesh as a candidate.
  // 2) Solve for the best rigid transform between the moving mesh and the target mesh using Procrustes analysis.
  // 3) Transform the moving mesh using the retrieved transform
  // 4) Loop to step 1 if the result is not aligned with the target (or if we didn’t reach the limit number of iterations)

  // non-rigid does the same thing, but instead of rigid fitting it does GP regression


  // Get some points
  val sampler = UniformMeshSampler3D(model.referenceMesh, numberOfPoints = 5000)
  val points : Seq[Point[_3D]] = sampler.sample.map(pointWithProbability => pointWithProbability._1) // we only want the points

  // These are better then points themselves
  val ptIds = points.map(point => model.referenceMesh.pointSet.findClosestPoint(point).id)

  // Get correspondence function
  def attributeCorrespondences(movingMesh: TriangleMesh[_3D], ptIds : Seq[PointId]) : Seq[(PointId, Point[_3D])] = {
    ptIds.map{ id : PointId =>
      val pt = movingMesh.pointSet.point(id)
      val closestPointOnMesh2 = targetMesh.pointSet.findClosestPoint(pt).point
      (id, closestPointOnMesh2)
    }
  }


  // get basic correspondences
  val correspondences = attributeCorrespondences(model.mean, ptIds)

  // need noise
  val littleNoise = MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3))

  // Now this function defines the GP regression. Get the posterior
  def fitModel(correspondences: Seq[(PointId, Point[_3D])]) : TriangleMesh[_3D] = {
    val regressionData = correspondences.map(correspondence =>
      (correspondence._1, correspondence._2, littleNoise)
    )
    val posterior = model.posterior(regressionData.toIndexedSeq)
    posterior.mean
  }

  // view results
  val fit = fitModel(correspondences)
  val resultGroup = ui.createGroup("results")
  val fitResultView = ui.show(resultGroup, fit, "fit")
  // We can see that the fit has been deformed to get a closer value

  // mow time for some iteration
  def nonrigidICP(movingMesh: TriangleMesh[_3D], ptIds : Seq[PointId], numberOfIterations : Int) : TriangleMesh[_3D] = {
    if (numberOfIterations == 0) movingMesh
    else {
      val correspondences = attributeCorrespondences(movingMesh, ptIds)
      val transformed = fitModel(correspondences)

      nonrigidICP(transformed, ptIds, numberOfIterations - 1)
    }
  }

  val finalFit = nonrigidICP( model.mean, ptIds, 20)

  ui.show(resultGroup, finalFit, "final fit")

}
