#!/usr/bin/env python3
"""
Smart Contract Deployment Script for Sepolia Testnet
Usage: python deploy_contract.py
"""

import json
import os
import requests
import time
from web3 import Web3
from eth_account import Account
from solcx import compile_source, install_solc
import sys

def compile_contract():
    """Compile the Solidity smart contract"""
    try:
        # Install solidity compiler if not available
        try:
            install_solc('0.8.19')
        except:
            pass

        # Read the contract source code
        with open('contracts/CommunityVoting.sol', 'r') as file:
            contract_source_code = file.read()

        # Compile the contract
        compiled_sol = compile_source(contract_source_code, solc_version='0.8.19')

        # Get contract interface
        contract_id, contract_interface = compiled_sol.popitem()

        return contract_interface['abi'], contract_interface['bin']

    except Exception as e:
        print(f"❌ Contract compilation failed: {e}")
        sys.exit(1)

def deploy_contract():
    """Deploy contract to Sepolia testnet"""
    print("🚀 Starting contract deployment to Sepolia testnet...")

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    # Blockchain configuration
    rpc_url = os.getenv('BLOCKCHAIN_RPC_URL')
    private_key = os.getenv('PRIVATE_KEY')
    chain_id = int(os.getenv('CHAIN_ID', 11155111))

    if not rpc_url or not private_key:
        print("❌ Missing BLOCKCHAIN_RPC_URL or PRIVATE_KEY in .env file")
        print("Please configure your .env file with:")
        print("BLOCKCHAIN_RPC_URL=https://sepolia.infura.io/v3/YOUR_INFURA_KEY")
        print("PRIVATE_KEY=your_private_key_here")
        sys.exit(1)

    if "YOUR_INFURA_KEY" in rpc_url or "YOUR_PRIVATE_KEY" in private_key:
        print("❌ Please replace placeholder values in .env file:")
        print("- Get Infura key from https://infura.io")
        print("- Export private key from MetaMask wallet")
        sys.exit(1)

    # Connect to blockchain
    try:
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not w3.is_connected():
            print("❌ Failed to connect to Sepolia network")
            sys.exit(1)
        print("✅ Connected to Sepolia network")
    except Exception as e:
        print(f"❌ Connection error: {e}")
        sys.exit(1)

    # Setup account
    account = Account.from_key(private_key)
    print(f"📋 Deploying from account: {account.address}")

    # Check balance
    balance = w3.eth.get_balance(account.address)
    balance_eth = w3.from_wei(balance, 'ether')
    print(f"💰 Account balance: {balance_eth:.4f} ETH")

    if balance_eth < 0.01:
        print("❌ Insufficient balance. You need at least 0.01 ETH for deployment.")
        print("Get Sepolia ETH from faucets:")
        print("- https://sepoliafaucet.com/")
        print("- https://faucet.sepolia.dev/")
        sys.exit(1)

    # Compile contract
    print("🔨 Compiling smart contract...")
    abi, bytecode = compile_contract()
    print("✅ Contract compiled successfully")

    # Create contract object
    contract = w3.eth.contract(abi=abi, bytecode=bytecode)

    # Build constructor transaction
    print("📄 Building deployment transaction...")
    constructor_txn = contract.constructor().build_transaction({
        'from': account.address,
        'gas': 3000000,
        'gasPrice': w3.to_wei('20', 'gwei'),
        'nonce': w3.eth.get_transaction_count(account.address),
        'chainId': chain_id
    })

    # Sign transaction
    signed_txn = w3.eth.account.sign_transaction(constructor_txn, private_key)

    # Send transaction
    print("🚀 Deploying contract...")
    tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
    print(f"📋 Transaction hash: {tx_hash.hex()}")

    # Wait for confirmation
    print("⏳ Waiting for transaction confirmation...")
    try:
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
    except Exception as e:
        print(f"❌ Transaction failed or timed out: {e}")
        sys.exit(1)

    if tx_receipt.status == 1:
        contract_address = tx_receipt.contractAddress
        print(f"✅ Contract deployed successfully!")
        print(f"📋 Contract address: {contract_address}")
        print(f"🔗 View on Etherscan: https://sepolia.etherscan.io/address/{contract_address}")
        print(f"⛽ Gas used: {tx_receipt.gasUsed:,}")

        # Save ABI and address
        deployment_info = {
            "address": contract_address,
            "abi": abi,
            "transaction_hash": tx_hash.hex(),
            "block_number": tx_receipt.blockNumber,
            "gas_used": tx_receipt.gasUsed
        }

        with open('contracts/deployment_info.json', 'w') as f:
            json.dump(deployment_info, f, indent=2)
        print("💾 Deployment info saved to contracts/deployment_info.json")

        # Update .env file
        update_env_file(contract_address)

        print("\n🎉 Deployment completed successfully!")

        # Ask for contract verification
        verify_choice = input("\n🔍 Would you like to verify the contract on Etherscan? (y/n): ").lower().strip()
        if verify_choice in ['y', 'yes']:
            verify_contract(contract_address, abi, bytecode)

        print("\nNext steps:")
        print("1. Restart your backend server")
        print("2. Test proposal creation through your API")
        print("3. Contract is ready for use!")

    else:
        print("❌ Contract deployment failed!")
        sys.exit(1)

def update_env_file(contract_address):
    """Update .env file with contract address"""
    try:
        # Read current .env
        with open('.env', 'r') as f:
            lines = f.readlines()

        # Update CONTRACT_ADDRESS line
        updated = False
        for i, line in enumerate(lines):
            if line.startswith('CONTRACT_ADDRESS='):
                lines[i] = f'CONTRACT_ADDRESS={contract_address}\n'
                updated = True
                break

        if not updated:
            lines.append(f'CONTRACT_ADDRESS={contract_address}\n')

        # Write back to .env
        with open('.env', 'w') as f:
            f.writelines(lines)

        print("✅ Updated .env file with contract address")

    except Exception as e:
        print(f"⚠️  Failed to update .env file: {e}")
        print(f"Please manually add: CONTRACT_ADDRESS={contract_address}")

def verify_contract(contract_address, abi, bytecode):
    """Verify contract on Etherscan"""
    print("\n🔍 Starting contract verification...")

    etherscan_api_key = os.getenv('ETHERSCAN_API_KEY')

    if not etherscan_api_key:
        print("⚠️  Skipping verification - ETHERSCAN_API_KEY not found in .env file")
        print("💡 Get a free API key from https://etherscan.io/apis and add to .env:")
        print("   ETHERSCAN_API_KEY=your_api_key_here")
        return

    try:
        # Read the contract source code
        with open('contracts/CommunityVoting.sol', 'r') as f:
            source_code = f.read()

        # Prepare verification data
        verification_data = {
            'apikey': etherscan_api_key,
            'module': 'contract',
            'action': 'verifysourcecode',
            'contractaddress': contract_address,
            'sourceCode': source_code,
            'codeformat': 'solidity-single-file',
            'contractname': 'CommunityVoting',
            'compilerversion': 'v0.8.19+commit.7dd6d404',
            'optimizationUsed': '0',
            'runs': '200',
            'constructorArguements': '',
            'evmversion': '',
            'licenseType': '3'  # MIT License
        }

        # Submit verification
        print("📤 Submitting verification request...")
        response = requests.post('https://api-sepolia.etherscan.io/api', data=verification_data)
        result = response.json()

        if result['status'] == '1':
            guid = result['result']
            print(f"✅ Verification submitted successfully!")
            print(f"📋 GUID: {guid}")

            # Check verification status
            print("⏳ Checking verification status...")
            for attempt in range(30):  # Wait up to 5 minutes
                time.sleep(10)

                status_response = requests.get('https://api-sepolia.etherscan.io/api', params={
                    'apikey': etherscan_api_key,
                    'module': 'contract',
                    'action': 'checkverifystatus',
                    'guid': guid
                })

                status_result = status_response.json()

                if status_result['status'] == '1':
                    if status_result['result'] == 'Pass - Verified':
                        print("✅ Contract verified successfully!")
                        print(f"🔗 View verified contract: https://sepolia.etherscan.io/address/{contract_address}#code")
                        break
                    elif status_result['result'] == 'Fail - Unable to verify':
                        print("❌ Verification failed")
                        print("This might be due to compiler version mismatch or other issues")
                        break
                    else:
                        print(f"⏳ Status: {status_result['result']}")
                else:
                    print(f"⏳ Checking... (attempt {attempt + 1}/30)")
            else:
                print("⚠️  Verification timeout - please check manually on Etherscan")
        else:
            print(f"❌ Verification submission failed: {result.get('result', 'Unknown error')}")

    except Exception as e:
        print(f"❌ Verification error: {e}")

if __name__ == "__main__":
    try:
        deploy_contract()
    except KeyboardInterrupt:
        print("\n❌ Deployment cancelled by user")
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        sys.exit(1)