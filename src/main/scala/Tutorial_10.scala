object Tutorial_10 extends App {
  /* Iterative Closest Points for rigid alignment
  * Fabio Fehr
  * 25 May 2020
  * */

  // Automatic alignment

  import scalismo.geometry._
  import scalismo.common._
  import scalismo.ui.api._
  import scalismo.mesh._
  import scalismo.registration.LandmarkRegistration
  import scalismo.io.{MeshIO}
  import scalismo.numerics.UniformMeshSampler3D
  import breeze.linalg.{DenseMatrix, DenseVector}

  scalismo.initialize()
  implicit val rng = scalismo.utils.Random(42)

  val ui = ScalismoUI()

  //  We start by loading and visualizing two meshes (not aligned)

  val mesh1 = MeshIO.readMesh(new java.io.File("datasets/Paola.ply")).get
  val group1 = ui.createGroup("Dataset 1")
  val mesh1View = ui.show(group1, mesh1, "mesh1")

  val mesh2 = MeshIO.readMesh(new java.io.File("datasets/323.ply")).get
  val group2 = ui.createGroup("Dataset 2")
  val mesh2View = ui.show(group2, mesh2, "mesh2")
  mesh2View.color = java.awt.Color.RED

  // Now we want to automatically align them. We get candidate correspondences by just getting the closest points
  val ptIds = (0 until mesh1.pointSet.numberOfPoints by 50).map(i => PointId(i))
  ui.show(group1, ptIds.map(id => mesh1.pointSet.point(id)), "selected")

  // define a function that just selects the closes point
  def attributeCorrespondences(movingMesh: TriangleMesh[_3D], ptIds : Seq[PointId]) : Seq[(Point[_3D], Point[_3D])] = {
    ptIds.map{ id : PointId =>
      val pt = movingMesh.pointSet.point(id)
      val closestPointOnMesh2 = mesh2.pointSet.findClosestPoint(pt).point
      (pt, closestPointOnMesh2)
    }
  }

  val correspondences = attributeCorrespondences(mesh1, ptIds)
  val targetPoints = correspondences.map(pointPair => pointPair._2)
  ui.show(group2, targetPoints.toIndexedSeq, "correspondences")

  // These are kak, but we can use use them to get a good transformation then do it again

  val rigidTrans =  LandmarkRegistration.rigid3DLandmarkRegistration(correspondences, center = Point3D(0, 0, 0))
  val transformed = mesh1.transform(rigidTrans)
  val alignedMeshView = ui.show(group1, transformed, "aligned?")
  alignedMeshView.color = java.awt.Color.GREEN

  // Okay nowwww we find the closest point
  val newCorrespondences = attributeCorrespondences(transformed, ptIds)
  val newClosestPoints = newCorrespondences.map(pointPair => pointPair._2)
  ui.show(group2, newClosestPoints.toIndexedSeq, "newCandidateCorr")
  val newRigidTransformation =
    LandmarkRegistration.rigid3DLandmarkRegistration(newCorrespondences, center = Point3D(0, 0, 0))
  val newTransformed = transformed.transform(newRigidTransformation)
  val alignedMeshView2 =  ui.show(group2, newTransformed, "aligned??")
  alignedMeshView2.color = java.awt.Color.BLUE

  // These are still clearly wrong, but better. Lets iterate now

  def ICPRigidAlign(movingMesh: TriangleMesh[_3D], ptIds : Seq[PointId], numberOfIterations : Int) : TriangleMesh[_3D] = {
    if (numberOfIterations == 0) movingMesh
    else {
      val correspondences = attributeCorrespondences(movingMesh, ptIds)
      val transform = LandmarkRegistration.rigid3DLandmarkRegistration(correspondences, center = Point(0, 0, 0))
      val transformed = movingMesh.transform(transform)

      ICPRigidAlign(transformed, ptIds, numberOfIterations - 1)
    }
  }

  // Iterate many times to get a good fit.
  val rigidfit = ICPRigidAlign(mesh1, ptIds, 150)
  val rigidFitView = ui.show(group1, rigidfit, "ICP_rigid_fit")
  rigidFitView.color = java.awt.Color.YELLOW

  // Now it fits nicely, but local mins are a problem so poor starting values are a lass
}
