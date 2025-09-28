// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract CommunityVoting {
    enum ProposalStatus { Active, Accepted, Declined }
    enum Vote { None, Yes, No }

    struct EnvironmentalData {
        uint256 ndviBefore;
        uint256 ndviAfter;
        uint256 pm25Before;
        uint256 pm25After;
        uint256 pm25IncreasePercent;
        uint256 vegetationLossPercent;
    }

    struct Demographics {
        uint256 children;
        uint256 adults;
        uint256 seniors;
        uint256 totalAffectedPopulation;
    }

    struct Proposal {
        uint256 id;
        string parkName;
        string parkId;
        string description;
        uint256 endDate;
        ProposalStatus status;
        uint256 yesVotes;
        uint256 noVotes;
        EnvironmentalData environmentalData;
        Demographics demographics;
        address creator;
        bool exists;
    }

    mapping(uint256 => Proposal) public proposals;
    mapping(uint256 => mapping(address => Vote)) public userVotes;
    mapping(uint256 => mapping(address => bool)) public hasVoted;

    uint256 public proposalCounter;
    address public owner;

    event ProposalCreated(
        uint256 indexed proposalId,
        string parkName,
        string parkId,
        uint256 endDate,
        address creator
    );

    event VoteCast(
        uint256 indexed proposalId,
        address indexed voter,
        Vote vote
    );

    event ProposalStatusUpdated(
        uint256 indexed proposalId,
        ProposalStatus newStatus
    );

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }

    modifier proposalExists(uint256 _proposalId) {
        require(proposals[_proposalId].exists, "Proposal does not exist");
        _;
    }

    modifier proposalActive(uint256 _proposalId) {
        require(proposals[_proposalId].status == ProposalStatus.Active, "Proposal is not active");
        require(block.timestamp <= proposals[_proposalId].endDate, "Voting period has ended");
        _;
    }

    modifier hasNotVoted(uint256 _proposalId) {
        require(!hasVoted[_proposalId][msg.sender], "User has already voted");
        _;
    }

    constructor() {
        owner = msg.sender;
        proposalCounter = 0;
    }

    function createProposal(
        string memory _parkName,
        string memory _parkId,
        string memory _description,
        uint256 _endDate,
        uint256[6] memory _environmentalData, // [ndviBefore, ndviAfter, pm25Before, pm25After, pm25IncreasePercent, vegetationLossPercent]
        uint256[4] memory _demographics // [children, adults, seniors, totalAffectedPopulation]
    ) external returns (uint256) {
        require(_endDate > block.timestamp, "End date must be in the future");
        require(bytes(_parkName).length > 0, "Park name cannot be empty");
        require(bytes(_parkId).length > 0, "Park ID cannot be empty");

        proposalCounter++;
        uint256 newProposalId = proposalCounter;

        proposals[newProposalId] = Proposal({
            id: newProposalId,
            parkName: _parkName,
            parkId: _parkId,
            description: _description,
            endDate: _endDate,
            status: ProposalStatus.Active,
            yesVotes: 0,
            noVotes: 0,
            environmentalData: EnvironmentalData({
                ndviBefore: _environmentalData[0],
                ndviAfter: _environmentalData[1],
                pm25Before: _environmentalData[2],
                pm25After: _environmentalData[3],
                pm25IncreasePercent: _environmentalData[4],
                vegetationLossPercent: _environmentalData[5]
            }),
            demographics: Demographics({
                children: _demographics[0],
                adults: _demographics[1],
                seniors: _demographics[2],
                totalAffectedPopulation: _demographics[3]
            }),
            creator: msg.sender,
            exists: true
        });

        emit ProposalCreated(newProposalId, _parkName, _parkId, _endDate, msg.sender);
        return newProposalId;
    }

    function vote(uint256 _proposalId, bool _vote) external
        proposalExists(_proposalId)
        proposalActive(_proposalId)
        hasNotVoted(_proposalId)
    {
        Vote voteChoice = _vote ? Vote.Yes : Vote.No;
        userVotes[_proposalId][msg.sender] = voteChoice;
        hasVoted[_proposalId][msg.sender] = true;

        if (_vote) {
            proposals[_proposalId].yesVotes++;
        } else {
            proposals[_proposalId].noVotes++;
        }

        emit VoteCast(_proposalId, msg.sender, voteChoice);
    }

    function updateProposalStatus(uint256 _proposalId) external
        proposalExists(_proposalId)
    {
        require(block.timestamp > proposals[_proposalId].endDate, "Voting period has not ended");
        require(proposals[_proposalId].status == ProposalStatus.Active, "Proposal is not active");

        if (proposals[_proposalId].yesVotes > proposals[_proposalId].noVotes) {
            proposals[_proposalId].status = ProposalStatus.Accepted;
        } else {
            proposals[_proposalId].status = ProposalStatus.Declined;
        }

        emit ProposalStatusUpdated(_proposalId, proposals[_proposalId].status);
    }

    function getProposal(uint256 _proposalId) external view
        proposalExists(_proposalId)
        returns (
            uint256 id,
            string memory parkName,
            string memory parkId,
            string memory description,
            uint256 endDate,
            ProposalStatus status,
            uint256 yesVotes,
            uint256 noVotes,
            address creator
        )
    {
        Proposal memory proposal = proposals[_proposalId];
        return (
            proposal.id,
            proposal.parkName,
            proposal.parkId,
            proposal.description,
            proposal.endDate,
            proposal.status,
            proposal.yesVotes,
            proposal.noVotes,
            proposal.creator
        );
    }

    function getEnvironmentalData(uint256 _proposalId) external view
        proposalExists(_proposalId)
        returns (
            uint256 ndviBefore,
            uint256 ndviAfter,
            uint256 pm25Before,
            uint256 pm25After,
            uint256 pm25IncreasePercent,
            uint256 vegetationLossPercent
        )
    {
        EnvironmentalData memory data = proposals[_proposalId].environmentalData;
        return (
            data.ndviBefore,
            data.ndviAfter,
            data.pm25Before,
            data.pm25After,
            data.pm25IncreasePercent,
            data.vegetationLossPercent
        );
    }

    function getDemographics(uint256 _proposalId) external view
        proposalExists(_proposalId)
        returns (
            uint256 children,
            uint256 adults,
            uint256 seniors,
            uint256 totalAffectedPopulation
        )
    {
        Demographics memory demo = proposals[_proposalId].demographics;
        return (
            demo.children,
            demo.adults,
            demo.seniors,
            demo.totalAffectedPopulation
        );
    }

    function getVoteCounts(uint256 _proposalId) external view
        proposalExists(_proposalId)
        returns (uint256 yesVotes, uint256 noVotes)
    {
        return (proposals[_proposalId].yesVotes, proposals[_proposalId].noVotes);
    }

    function getUserVote(uint256 _proposalId, address _user) external view
        proposalExists(_proposalId)
        returns (Vote)
    {
        return userVotes[_proposalId][_user];
    }

    function hasUserVoted(uint256 _proposalId, address _user) external view
        proposalExists(_proposalId)
        returns (bool)
    {
        return hasVoted[_proposalId][_user];
    }

    function isProposalActive(uint256 _proposalId) external view
        proposalExists(_proposalId)
        returns (bool)
    {
        return proposals[_proposalId].status == ProposalStatus.Active &&
               block.timestamp <= proposals[_proposalId].endDate;
    }

    function getAllActiveProposals() external view returns (uint256[] memory) {
        uint256[] memory activeProposals = new uint256[](proposalCounter);
        uint256 activeCount = 0;

        for (uint256 i = 1; i <= proposalCounter; i++) {
            if (proposals[i].exists && proposals[i].status == ProposalStatus.Active) {
                activeProposals[activeCount] = i;
                activeCount++;
            }
        }

        // Resize array to actual count
        uint256[] memory result = new uint256[](activeCount);
        for (uint256 i = 0; i < activeCount; i++) {
            result[i] = activeProposals[i];
        }

        return result;
    }

    function getAllClosedProposals() external view returns (uint256[] memory) {
        uint256[] memory closedProposals = new uint256[](proposalCounter);
        uint256 closedCount = 0;

        for (uint256 i = 1; i <= proposalCounter; i++) {
            if (proposals[i].exists &&
                (proposals[i].status == ProposalStatus.Accepted || proposals[i].status == ProposalStatus.Declined)) {
                closedProposals[closedCount] = i;
                closedCount++;
            }
        }

        // Resize array to actual count
        uint256[] memory result = new uint256[](closedCount);
        for (uint256 i = 0; i < closedCount; i++) {
            result[i] = closedProposals[i];
        }

        return result;
    }

    function getTotalProposals() external view returns (uint256) {
        return proposalCounter;
    }
}