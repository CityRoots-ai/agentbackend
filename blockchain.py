from web3 import Web3
from eth_account import Account
import json
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BlockchainService:
    def __init__(self):
        # Sepolia testnet configuration
        self.w3 = Web3(Web3.HTTPProvider(os.getenv('BLOCKCHAIN_RPC_URL', 'https://sepolia.infura.io/v3/YOUR_INFURA_KEY')))
        self.private_key = os.getenv('PRIVATE_KEY')
        if self.private_key:
            self.account = Account.from_key(self.private_key)
        self.contract_address = os.getenv('CONTRACT_ADDRESS')
        self.chain_id = int(os.getenv('CHAIN_ID', 11155111))  # Sepolia chain ID
        self.explorer_base_url = os.getenv('EXPLORER_BASE_URL', 'https://sepolia.etherscan.io')

        # Contract ABI - Complete ABI for CommunityVoting contract
        self.contract_abi = [
            {
                "inputs": [],
                "stateMutability": "nonpayable",
                "type": "constructor"
            },
            {
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "internalType": "uint256", "name": "proposalId", "type": "uint256"},
                    {"indexed": False, "internalType": "string", "name": "parkName", "type": "string"},
                    {"indexed": False, "internalType": "string", "name": "parkId", "type": "string"},
                    {"indexed": False, "internalType": "uint256", "name": "endDate", "type": "uint256"},
                    {"indexed": False, "internalType": "address", "name": "creator", "type": "address"}
                ],
                "name": "ProposalCreated",
                "type": "event"
            },
            {
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "internalType": "uint256", "name": "proposalId", "type": "uint256"},
                    {"indexed": False, "internalType": "enum CommunityVoting.ProposalStatus", "name": "newStatus", "type": "uint8"}
                ],
                "name": "ProposalStatusUpdated",
                "type": "event"
            },
            {
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "internalType": "uint256", "name": "proposalId", "type": "uint256"},
                    {"indexed": True, "internalType": "address", "name": "voter", "type": "address"},
                    {"indexed": False, "internalType": "enum CommunityVoting.Vote", "name": "vote", "type": "uint8"}
                ],
                "name": "VoteCast",
                "type": "event"
            },
            {
                "inputs": [
                    {"internalType": "string", "name": "_parkName", "type": "string"},
                    {"internalType": "string", "name": "_parkId", "type": "string"},
                    {"internalType": "string", "name": "_description", "type": "string"},
                    {"internalType": "uint256", "name": "_endDate", "type": "uint256"},
                    {"internalType": "uint256[6]", "name": "_environmentalData", "type": "uint256[6]"},
                    {"internalType": "uint256[4]", "name": "_demographics", "type": "uint256[4]"}
                ],
                "name": "createProposal",
                "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {"internalType": "uint256", "name": "_proposalId", "type": "uint256"}
                ],
                "name": "getProposal",
                "outputs": [
                    {"internalType": "uint256", "name": "id", "type": "uint256"},
                    {"internalType": "string", "name": "parkName", "type": "string"},
                    {"internalType": "string", "name": "parkId", "type": "string"},
                    {"internalType": "string", "name": "description", "type": "string"},
                    {"internalType": "uint256", "name": "endDate", "type": "uint256"},
                    {"internalType": "enum CommunityVoting.ProposalStatus", "name": "status", "type": "uint8"},
                    {"internalType": "uint256", "name": "yesVotes", "type": "uint256"},
                    {"internalType": "uint256", "name": "noVotes", "type": "uint256"},
                    {"internalType": "address", "name": "creator", "type": "address"}
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [
                    {"internalType": "uint256", "name": "_proposalId", "type": "uint256"}
                ],
                "name": "getVoteCounts",
                "outputs": [
                    {"internalType": "uint256", "name": "yesVotes", "type": "uint256"},
                    {"internalType": "uint256", "name": "noVotes", "type": "uint256"}
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [
                    {"internalType": "uint256", "name": "_proposalId", "type": "uint256"},
                    {"internalType": "bool", "name": "_vote", "type": "bool"}
                ],
                "name": "vote",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "getTotalProposals",
                "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]

        if self.contract_address and self.contract_abi:
            self.contract = self.w3.eth.contract(
                address=self.contract_address,
                abi=self.contract_abi
            )

    def is_connected(self):
        """Check if connected to blockchain"""
        try:
            return self.w3.is_connected()
        except:
            return False

    def get_balance(self):
        """Get account balance in ETH"""
        if not self.account:
            return 0
        balance_wei = self.w3.eth.get_balance(self.account.address)
        return self.w3.from_wei(balance_wei, 'ether')

    async def create_proposal_on_blockchain(self, proposal_data):
        """Create proposal on smart contract"""
        try:
            if not self.is_connected():
                return {'success': False, 'error': 'Not connected to blockchain'}

            if not self.contract_address:
                return {'success': False, 'error': 'Contract address not configured'}

            if not self.account:
                return {'success': False, 'error': 'Private key not configured'}

            # Convert datetime string to timestamp
            end_date_str = proposal_data['endDate']
            # Parse different date formats
            try:
                end_timestamp = int(datetime.strptime(end_date_str, "%B %d, %Y").timestamp())
            except:
                try:
                    end_timestamp = int(datetime.strptime(end_date_str, "%B %d, %Y").timestamp())
                except:
                    # Default to 30 days from now if parsing fails
                    end_timestamp = int((datetime.now().timestamp() + 30 * 24 * 3600))

            analysis_data = proposal_data['analysisData']

            # Prepare environmental data array - scale values for blockchain storage
            ndvi_before = float(analysis_data.get('ndviBefore', 0))
            ndvi_after = float(analysis_data.get('ndviAfter', 0))
            pm25_before = float(analysis_data.get('pm25Before', 0))
            pm25_after = float(analysis_data.get('pm25After', 0))
            pm25_increase = float(analysis_data.get('pm25IncreasePercent', 0))
            vegetation_loss = float((ndvi_before - ndvi_after) * 100) if ndvi_before and ndvi_after else 0

            env_data = [
                int(ndvi_before * 10000),  # Scale for precision
                int(ndvi_after * 10000),
                int(pm25_before * 100),
                int(pm25_after * 100),
                int(pm25_increase * 100),
                int(vegetation_loss * 100)
            ]

            # Prepare demographics array
            demographics = analysis_data.get('demographics', {})
            demo_data = [
                int(demographics.get('kids', 0)),
                int(demographics.get('adults', 0)),
                int(demographics.get('seniors', 0)),
                int(analysis_data.get('affectedPopulation10MinWalk', 0))
            ]

            logger.info(f"Environmental data: {env_data}")
            logger.info(f"Demographics data: {demo_data}")

            # Check balance
            balance = self.get_balance()
            if balance < 0.01:  # Need at least 0.01 ETH for gas
                return {'success': False, 'error': f'Insufficient balance: {balance} ETH'}

            # Generate blockchain-optimized summary using Gemini
            description = await self._generate_blockchain_summary(
                proposal_data['proposalSummary'],
                analysis_data
            )

            # Estimate gas first
            try:
                estimated_gas = self.contract.functions.createProposal(
                    proposal_data['parkName'],
                    proposal_data['parkId'],
                    description,
                    end_timestamp,
                    env_data,
                    demo_data
                ).estimate_gas({'from': self.account.address})

                gas_limit = int(estimated_gas * 1.2)  # Add 20% buffer
                logger.info(f"Estimated gas: {estimated_gas}, Using: {gas_limit}")

            except Exception as e:
                logger.warning(f"Gas estimation failed: {e}, using default")
                gas_limit = 1000000

            transaction = self.contract.functions.createProposal(
                proposal_data['parkName'],
                proposal_data['parkId'],
                description,
                end_timestamp,
                env_data,
                demo_data
            ).build_transaction({
                'from': self.account.address,
                'gas': gas_limit,
                'gasPrice': self.w3.to_wei('10', 'gwei'),  # Lower gas price
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'chainId': self.chain_id
            })

            # Sign and send transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.raw_transaction)

            logger.info(f"Transaction sent: {tx_hash.hex()}")

            # Wait for confirmation (with timeout)
            try:
                tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Transaction timeout or failed: {str(e)}',
                    'transaction_hash': tx_hash.hex()
                }

            # Get proposal ID from logs
            proposal_id = None
            try:
                if tx_receipt.logs and tx_receipt.status == 1:
                    # Try to decode logs
                    decoded_logs = self.contract.events.ProposalCreated().process_receipt(tx_receipt)
                    if decoded_logs:
                        proposal_id = decoded_logs[0]['args']['proposalId']
                        logger.info(f"Successfully extracted proposal ID: {proposal_id}")
                    else:
                        # Fallback: get total proposals count
                        try:
                            proposal_id = self.contract.functions.getTotalProposals().call()
                            logger.info(f"Using total proposals as ID: {proposal_id}")
                        except:
                            proposal_id = "Unknown"
                else:
                    logger.warning("Transaction failed or no logs found")
            except Exception as e:
                logger.warning(f"Could not decode proposal ID from logs: {e}")
                # Fallback to getting total proposals
                try:
                    proposal_id = self.contract.functions.getTotalProposals().call()
                except:
                    proposal_id = "Unknown"

            return {
                'success': True,
                'transaction_hash': tx_hash.hex(),
                'proposal_id': proposal_id,
                'block_number': tx_receipt.blockNumber,
                'gas_used': tx_receipt.gasUsed,
                'explorer_url': f"{self.explorer_base_url}/tx/{tx_hash.hex()}"
            }

        except Exception as e:
            logger.error(f"Blockchain transaction failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_proposal_from_blockchain(self, proposal_id):
        """Get proposal data from blockchain"""
        try:
            if not self.contract:
                return {'success': False, 'error': 'Contract not initialized'}

            proposal = self.contract.functions.getProposal(proposal_id).call()
            vote_counts = self.contract.functions.getVoteCounts(proposal_id).call()

            return {
                'success': True,
                'proposal': {
                    'id': proposal[0],
                    'parkName': proposal[1],
                    'parkId': proposal[2],
                    'description': proposal[3],
                    'endDate': proposal[4],
                    'status': proposal[5],
                    'yesVotes': vote_counts[0],
                    'noVotes': vote_counts[1],
                    'creator': proposal[8]
                }
            }
        except Exception as e:
            logger.error(f"Error fetching proposal: {str(e)}")
            return {'success': False, 'error': str(e)}

    async def _generate_blockchain_summary(self, full_summary, analysis_data):
        """Generate a concise summary for blockchain storage using Gemini"""
        try:
            from google import genai
            import os

            # Initialize Gemini client
            client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

            # Create prompt for summary generation
            prompt = f"""Create a neutral data summary for a park proposal focusing only on NDVI and PM2.5 metrics.

Key data points to include:
- Park name: {analysis_data.get('parkName', 'Unknown')}
- NDVI change: {analysis_data.get('ndviBefore', 0)} → {analysis_data.get('ndviAfter', 0)}
- PM2.5 increase: {analysis_data.get('pm25IncreasePercent', 0)}%

Requirements:
- Must be between 230-240 characters exactly
- Only include NDVI and PM2.5 data
- Neutral factual tone only
- No emotional words or judgments
- Include exact numerical values
- Fill remaining space with relevant environmental context

Format: Park [name]: Environmental analysis shows NDVI decline from [before] to [after] and PM2.5 increase of [percent]% indicating vegetation loss and air quality degradation in the area.

Return only the factual summary that is exactly 230-240 characters."""

            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt
            )

            summary = response.text.strip()

            # Ensure it's within the strict 230-240 character range
            if len(summary) < 230:
                # Pad with environmental context if too short
                padding_needed = 230 - len(summary)
                environmental_context = " Environmental impact assessment indicates significant changes to local ecosystem conditions."
                summary += environmental_context[:padding_needed]
            elif len(summary) > 240:
                # Trim if too long
                summary = summary[:240]

            logger.info(f"Generated blockchain summary ({len(summary)} chars): {summary}")
            return summary

        except Exception as e:
            logger.warning(f"Failed to generate Gemini summary: {e}")
            # Fallback to neutral factual summary focusing only on NDVI and PM2.5
            park_name = analysis_data.get('parkName', 'Park')
            ndvi_before = analysis_data.get('ndviBefore', 0)
            ndvi_after = analysis_data.get('ndviAfter', 0)
            pm25_increase = analysis_data.get('pm25IncreasePercent', 0)

            fallback = f"{park_name}: Environmental analysis shows NDVI decline from {ndvi_before} to {ndvi_after} and PM2.5 increase of {pm25_increase}% indicating vegetation loss and air quality degradation in the analyzed area."

            # Ensure fallback is also 230-240 characters
            if len(fallback) < 230:
                fallback += " Impact assessment completed."
            elif len(fallback) > 240:
                fallback = fallback[:240]

            return fallback