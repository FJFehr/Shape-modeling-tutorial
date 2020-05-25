object Tutorial_14 extends App  {

  /* Model fitting using MCMC
  * https://scalismo.org/tutorials/tutorial14.html
  * Bayesian model fitting using Markov Chain Monte Carlo in Scalismo.
  * This is a toy example not done for shape modeling
  * */

  // Bayesian Problem: We are trying to fit a (univariate) normal distribution ,
  // with unknown mean and unknown standard deviation to a set of data points.
  // We compute a posterior!

  // Posterior = (Likelihood * Prior) / marginal likelihood

  // shape model parameters, and the data y are not simulated numbers,
  // but measurements of the target object, such as a set of landmark points, a surface or even an image.


  // Metropolis Hastings Algorithm
  // How do we approach this fitting? Use MH to draw samples from any distribution,
  // given that the unnormalized distribution can be evaluated point-wise

  //For setting up the Metropolis-Hastings algorithm, we need two things:
  // 1) The (unnormalized) target distribution, from which we want to sample.
  //    In our case this is the posterior distribution p(θ∣y).
  //    In Scalismo the corresponding class is called the Distribution Evaluator.
  // 2) A proposal distribution Q(θ′∣θ), which generates for a given sample θ a new sample θ′.

  // MH uses accepting and rejecting based on the prob under the target density
  // It uses the proposal generator to perturb a given sample θ to obtain a new sample θ′.
  // Then it checks, using the evaluator, which of the two samples, θ or θ′ is more likely
  // and uses this ratio as a basis for rejecting or accepting the new sample.

  import scalismo.sampling.algorithms.MetropolisHastings
  import scalismo.sampling.evaluators.ProductEvaluator
  import scalismo.sampling.loggers.AcceptRejectLogger
  import scalismo.sampling.proposals.MixtureProposal
  import scalismo.sampling.{DistributionEvaluator, ProposalGenerator, TransitionProbability}
  scalismo.initialize()
  implicit val rng = scalismo.utils.Random(42)

  // To test our method, we generate data from a normal distribution N(−5,17).

  val mu = -5
  val sigma = 17

  val trueDistribution = breeze.stats.distributions.Gaussian(mu, sigma)
  val data = for (_ <- 0 until 100) yield {
    trueDistribution.draw()
  }

  // Theta
  case class Parameters(mu : Double, sigma: Double)

  // sample from the chain.
  case class Sample(parameters : Parameters, generatedBy : String)

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Evaluators: Modelling the target density /////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // subclass of the class DistributionEvalutor = log probability of a sample.
  // MUST COMMENT THESE TRAITS OUT AS THEY ARE DEFINED

//  trait DistributionEvaluator[A] {
//    /** log probability/density of sample */
//    def logValue(sample: A): Double
//  }

  // Now define evaluators for prior, likelihood assuming independence we can just write the likelihood or joint liklihood
  // as the product

  case class LikelihoodEvaluator(data : Seq[Double]) extends DistributionEvaluator[Sample] {

    // takes in theta which has the parms mu and sigma
    override def logValue(theta: Sample): Double = {
      // get likelihood
      val likelihood = breeze.stats.distributions.Gaussian(
        theta.parameters.mu, theta.parameters.sigma
      )
      // Get log likelihood
      val likelihoods = for (x <- data) yield {
        likelihood.logPdf(x)
      }
      // Sum it up
      likelihoods.sum
    }
  }

  // Now for the prior

  object PriorEvaluator extends DistributionEvaluator[Sample] {

    // Our prior mu is a RV with mean 0 and variance 20
    val priorDistMu = breeze.stats.distributions.Gaussian(0, 20)
    // Our prior sigma is a RV with mean 0 and variance 100
    val priorDistSigma = breeze.stats.distributions.Gaussian(0, 100)

    override def logValue(theta: Sample): Double = {
      priorDistMu.logPdf(theta.parameters.mu)
      + priorDistSigma.logPdf(theta.parameters.sigma)
    }
  }

  // Now we product prior and likelihood
  val posteriorEvaluator = ProductEvaluator(PriorEvaluator, LikelihoodEvaluator(data))

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // The proposal generator ///////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//  trait ProposalGenerator[A] {
//    /** draw a sample from this proposal distribution, may depend on current state */
//    def propose(current: A): A
//  }

  // To use a proposal generator in the MH algorithm, we also need to implement the trait TransitionProbability:

//  trait TransitionProbability[A] extends TransitionRatio[A] {
//    /** rate of transition from to (log value) */
//    def logTransitionProbability(from: A, to: A): Double
//  }

  // random walk proposal.  taking a step of random length in a random direction.
  // The second argument allows for seeds and reproducibility Ie set a seed and we can not reproduce

  case class RandomWalkProposal(stddevMu: Double, stddevSigma : Double)(implicit rng : scalismo.utils.Random)
    extends ProposalGenerator[Sample] with TransitionProbability[Sample] {

    override def propose(sample: Sample): Sample = {
      val newParameters = Parameters(
        mu = sample.parameters.mu + rng.breezeRandBasis.gaussian(0, stddevMu).draw(),
        sigma = sample.parameters.sigma + rng.breezeRandBasis.gaussian(0, stddevSigma).draw()
      )

      Sample(newParameters, s"randomWalkProposal ($stddevMu, $stddevSigma)")
    }

    override def logTransitionProbability(from: Sample, to: Sample) : Double = {

      val stepDistMu = breeze.stats.distributions.Gaussian(0, stddevMu)
      val stepDistSigma = breeze.stats.distributions.Gaussian(0, stddevSigma)

      val residualMu = to.parameters.mu - from.parameters.mu
      val residualSigma = to.parameters.sigma - from.parameters.sigma
      stepDistMu.logPdf(residualMu)  + stepDistMu.logPdf(residualSigma)
    }
  }

  // These parms are stdevs for means and variances. Meaning how variable are my parameters
  val smallStepProposal = RandomWalkProposal(3.0, 1.0)
  val largeStepProposal = RandomWalkProposal(9.0, 3.0)

  // Exploitation vs exploration!
  // MixtureProposal, which chooses the individual proposals with a given probability.
  // Here We choose to take the large step 20% of the time, and the smaller steps 80% of the time:

  val generator = MixtureProposal.fromProposalsWithTransition[Sample](
    (0.8, smallStepProposal),
    (0.2, largeStepProposal)
  )

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Building the Markov Chain  ///////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  val chain = MetropolisHastings(generator, posteriorEvaluator)

  // To run the chain, we obtain an iterator, which we then consume to drive the sampling generation.
  // To obtain the iterator, we need to specify the initial sample: should get to (-5, 17)

  val initialSample = Sample(Parameters(0.0, 10.0), generatedBy="initial")
  val mhIterator = chain.iterator(initialSample)

  // Our initial parameters might be far away from a high-probability area of our target density. Thus MANY Samples
  // We therefore have to drop the samples in this burn-in phase, before we use the samples:

  val samples = mhIterator.drop(1000).take(5000).toIndexedSeq

  val estimatedMean = samples.map(sample => sample.parameters.mu).sum  / samples.size
  // estimatedMean: Double = -6.623743279380656
  println("estimated mean is " + estimatedMean)
  // estimated mean is -6.623743279380656
  val estimatedSigma = samples.map(sample => sample.parameters.sigma).sum / samples.size
  // estimatedSigma: Double = 19.656746075626078
  println("estimated sigma is " + estimatedSigma)
  // estimated sigma is 19.656746075626078
  // VERY HIGH VARIATION ON MULTIPLE RUNS

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Debugging the markov Chain ///////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // if its broke then its probably that our proposals are not suitable for the target distribution.
  // Define a logger. To write a logger, we need to extend the trait AcceptRejectLogger, which is defined as follows:

//  trait AcceptRejectLogger[A] {
//    def accept(current: A, sample: A, generator: ProposalGenerator[A], evaluator: DistributionEvaluator[A]): Unit
//
//    def reject(current: A, sample: A, generator: ProposalGenerator[A], evaluator: DistributionEvaluator[A]): Unit
//  }

  // The two methods, accept and reject are called whenever a sample is accepted or rejected.
  //  We can overwrite these methods to implement our debugging code.
  // This calculates the acceptance ratio (Which helps us tell if the proposal is suitable)

  class Logger extends AcceptRejectLogger[Sample] {
    private val numAccepted = collection.mutable.Map[String, Int]()
    private val numRejected = collection.mutable.Map[String, Int]()

    override def accept(current: Sample,
                        sample: Sample,
                        generator: ProposalGenerator[Sample],
                        evaluator: DistributionEvaluator[Sample]
                       ): Unit = {
      val numAcceptedSoFar = numAccepted.getOrElseUpdate(sample.generatedBy, 0)
      numAccepted.update(sample.generatedBy, numAcceptedSoFar + 1)
    }

    override def reject(current: Sample,
                        sample: Sample,
                        generator: ProposalGenerator[Sample],
                        evaluator: DistributionEvaluator[Sample]
                       ): Unit = {
      val numRejectedSoFar = numRejected.getOrElseUpdate(sample.generatedBy, 0)
      numRejected.update(sample.generatedBy, numRejectedSoFar + 1)
    }


    def acceptanceRatios() : Map[String, Double] = {
      val generatorNames = numRejected.keys.toSet.union(numAccepted.keys.toSet)
      val acceptanceRatios = for (generatorName <- generatorNames ) yield {
        val total = (numAccepted.getOrElse(generatorName, 0)
          + numRejected.getOrElse(generatorName, 0)).toDouble
        (generatorName, numAccepted.getOrElse(generatorName, 0) / total)
      }
      acceptanceRatios.toMap
    }
  }

  val logger = new Logger()
  val mhIteratorWithLogging = chain.iterator(initialSample, logger)
  val samples2 = mhIteratorWithLogging.drop(1000).take(3000).toIndexedSeq

  println("acceptance ratio is " +logger.acceptanceRatios())
  // acceptance ratio is Map(randomWalkProposal (3.0, 1.0) -> 0.5104496516782774, randomWalkProposal (9.0, 3.0) -> 0.15219976218787157)

  //  We see that the acceptance ratio of the random walk proposal,
  //  which takes the smaller step is quite high, but that the larger step is often rejected.
  //  We might therefore want to reduce this step size slightly,
  //  as a proposal that is so often rejected is not very efficient.

  println("fin")

}
